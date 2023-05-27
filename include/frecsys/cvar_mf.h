// Copyright 2023 Riku Togashi
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and
//
// limitations under the License.
#pragma once

#include <algorithm>
#include <chrono>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include "Eigen/Core"
#include "Eigen/Dense"

#include "frecsys/recommender.h"

namespace frecsys {

class CVaRMFRecommender : public Recommender {
public:
  CVaRMFRecommender(int embedding_dim, int num_users, int num_items,
                    float reg, float unobserved_weight, float alpha,
                    float stepsize, float stdev)
      : user_embedding_(num_users, embedding_dim),
        item_embedding_(num_items, embedding_dim),
        item_gramian_(embedding_dim, embedding_dim),
        dual_weight_(num_users), user_loss_(num_users),
        user_history_size_(num_users), item_reg_(num_items) {
    // Initialize embedding matrices
    float adjusted_stdev = stdev / sqrt(embedding_dim);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    init_matrix(&user_embedding_, gen, adjusted_stdev);
    init_matrix(&item_embedding_, gen, adjusted_stdev);
    item_gramian_ = item_embedding_.transpose() * item_embedding_;
    dual_weight_ = VectorXf::Ones(num_users) * alpha;
    user_loss_ = VectorXf::Zero(num_users);
    user_history_size_ = VectorXf::Zero(num_users);
    item_reg_ = VectorXf::Zero(num_items);

    regularization_ = reg;
    embedding_dim_ = embedding_dim;
    unobserved_weight_ = unobserved_weight;
    alpha_ = alpha;
    prev_xi_ = 0.0;

    stepsize_ = stepsize;
  }

  VectorXf Score(const int user_id,
                 const SpVector& user_history) override {
    throw("Function 'Score' is not implemented");
  }

  inline static const float ComputeLoss(
      const SpVector& user_history, const VectorXf& user_emb,
      const MatrixXf& item_embeddings, const MatrixXf& item_gramian,
      const float beta) {
    int num_items = item_embeddings.rows();
    float loss = 0;
    for (const auto& item_and_rating_index : user_history) {
      const int cp = item_and_rating_index.first;
      const VectorXf cp_v = item_embeddings.row(cp);
      loss += pow(cp_v.transpose() * user_emb - 1, 2.0);
    }
    loss /= user_history.size();
    float ireg = user_emb.transpose() * item_gramian * user_emb;
    loss += beta * ireg;
    loss /= 2.0;
    return loss;
  }

  inline static const VectorXf ProjectU(
      const SpVector& user_history, const VectorXf& user_emb,
      const MatrixXf& item_embeddings, const MatrixXf& gramian,
      const float reg, const float unobserved_weight,
      const float stepsize, const float weight) {
    int history_size = user_history.size();
    assert(history_size > 0);

    int embedding_dim = item_embeddings.cols();
    assert(embedding_dim > 0);

    MatrixXf matrix = MatrixXf::Zero(embedding_dim, embedding_dim);
    const int kMaxBatchSize = 128;
    auto matrix_symm = matrix.selfadjointView<Eigen::Lower>();
    VectorXf rhs = VectorXf::Zero(embedding_dim);
    const int batch_size = std::min(
        static_cast<int>(user_history.size()), kMaxBatchSize);
    int num_batched = 0;
    MatrixXf factor_batch(embedding_dim, batch_size);
    for (const auto& item_and_rating_index : user_history) {
      const int cp = item_and_rating_index.first;
      assert(cp < item_embeddings.rows());
      const VectorXf cp_v = item_embeddings.row(cp);
      rhs.noalias() += cp_v;

      factor_batch.col(num_batched).noalias() = cp_v;

      ++num_batched;
      if (num_batched == batch_size) {
        matrix_symm.rankUpdate(factor_batch);
        num_batched = 0;
      }
    }
    if (num_batched != 0) {
      auto factor_block =
          factor_batch.block(0, 0, embedding_dim, num_batched);
      matrix_symm.rankUpdate(factor_block);
    }
    matrix /= history_size;
    matrix.noalias() += unobserved_weight * gramian;
    matrix *= weight;
    rhs *= weight / history_size;
    for (int i = 0; i < embedding_dim; ++i) {
      matrix(i, i) += reg;
    }
    return user_emb - stepsize * (matrix * user_emb - rhs);
  }

  inline static const VectorXf ProjectV(
      const SpVector& item_history, const VectorXf& item_emb,
      const MatrixXf& user_embeddings, const MatrixXf& gramian,
      const float reg, const float unobserved_weight,
      const VectorXf& dual_weight, float stepsize) {
    assert(item_history.size() > 0);

    int embedding_dim = user_embeddings.cols();
    assert(embedding_dim > 0);

    const int kMaxBatchSize = 128;
    MatrixXf matrix = unobserved_weight * gramian;
    auto matrix_symm = matrix.selfadjointView<Eigen::Lower>();
    VectorXf rhs = VectorXf::Zero(embedding_dim);
    const int batch_size = std::min(
        static_cast<int>(item_history.size()), kMaxBatchSize);
    int num_batched = 0;
    MatrixXf factor_batch(embedding_dim, batch_size);
    for (const auto& user_and_rating_index : item_history) {
      const int cp = user_and_rating_index.first;
      assert(cp < user_embeddings.rows());
      const VectorXf cp_v = user_embeddings.row(cp);
      const float w = dual_weight(cp);
      rhs.noalias() += w * cp_v;

      factor_batch.col(num_batched).noalias() = sqrt(w) * cp_v;

      ++num_batched;
      if (num_batched == batch_size) {
        matrix_symm.rankUpdate(factor_batch);
        num_batched = 0;
      }
    }
    if (num_batched != 0) {
      auto factor_block =
          factor_batch.block(0, 0, embedding_dim, num_batched);
      matrix_symm.rankUpdate(factor_batch);
    }

    for (int i = 0; i < embedding_dim; ++i) {
      matrix(i, i) += reg;
    }

    return item_emb - stepsize * (matrix * item_emb - rhs);
  }

  inline static const VectorXf ProjectU_eval(
      const SpVector& user_history, const MatrixXf& item_embeddings,
      const MatrixXf& gramian, const float reg,
      const float unobserved_weight, const float weight) {
    int history_size = user_history.size();
    assert(history_size > 0);

    int embedding_dim = item_embeddings.cols();
    assert(embedding_dim > 0);

    MatrixXf matrix = MatrixXf::Zero(embedding_dim, embedding_dim);
    const int kMaxBatchSize = 128;
    auto matrix_symm = matrix.selfadjointView<Eigen::Lower>();
    VectorXf rhs = VectorXf::Zero(embedding_dim);
    const int batch_size = std::min(
        static_cast<int>(user_history.size()), kMaxBatchSize);
    int num_batched = 0;
    MatrixXf factor_batch(embedding_dim, batch_size);
    for (const auto& item_and_rating_index : user_history) {
      const int cp = item_and_rating_index.first;
      assert(cp < item_embeddings.rows());
      const VectorXf cp_v = item_embeddings.row(cp);
      rhs.noalias() += cp_v;

      factor_batch.col(num_batched).noalias() = cp_v;

      ++num_batched;
      if (num_batched == batch_size) {
        matrix_symm.rankUpdate(factor_batch);
        num_batched = 0;
      }
    }
    if (num_batched != 0) {
      auto factor_block =
          factor_batch.block(0, 0, embedding_dim, num_batched);
      matrix_symm.rankUpdate(factor_block);
    }
    matrix /= history_size;
    matrix.noalias() += unobserved_weight * gramian;
    matrix *= weight;
    rhs *= weight / history_size;
    for (int i = 0; i < embedding_dim; ++i) {
      matrix(i, i) += reg;
    }
    Eigen::LLT<MatrixXf, Eigen::Lower> cholesky(matrix);
    assert(cholesky.info() == Eigen::Success);
    return cholesky.solve(rhs);
  }

  // Custom implementation of EvaluateDataset that does the projection
  // using the iterative optimization algorithm.
  EvaluationResult EvaluateDataset(
      const VectorXi& k_list, const VectorXf& alpha_list,
      const Dataset& data, const SpMatrix& eval_by_user) override {
    std::unordered_map<int, int> user_to_ind;
    VectorXf prediction(data.num_tuples());
    MatrixXf user_embedding =
        MatrixXf::Zero(data.by_user().size(), embedding_dim_);

    // Initialize the user and predictions to 0.0. (Note: this code
    // needs to change if the embeddings would have biases).
    int num_users = 0;
    for (const auto& user_and_history : data.by_user()) {
      user_to_ind[user_and_history.first] = num_users;
      for (const auto& item_and_rating_index :
           user_and_history.second) {
        prediction.coeffRef(item_and_rating_index.second) = 0.0;
      }
      num_users++;
    }

    // Reproject the users.
    VectorXf dual_weight = VectorXf::Ones(num_users);
    StepU_eval(
        data.by_user(),
        [&](const int user_id) -> MatrixXf::RowXpr {
          return user_embedding.row(user_to_ind[user_id]);
        },
        item_embedding_, item_gramian_,
        [&](const int user_id) -> float {
          return dual_weight(user_to_ind[user_id]);
        });

    // Evaluate the dataset.
    int num_items = item_embedding_.rows();
    return EvaluateDatasetInternal(
        num_items, k_list, alpha_list, user_to_ind, data,
        eval_by_user,
        [&](const int user_id, const SpVector& history) -> VectorXf {
          return item_embedding_ *
                 user_embedding.row(user_to_ind[user_id]).transpose();
        });
  }

  void Train(const Dataset& data) override {
    PrintLosses(data);

    float residual_z = ComputeUserWeights(
        data.by_user(), dual_weight_, user_loss_, prev_xi_);

    MatrixXf user_embedding_prev = user_embedding_;
    float residual_U = StepU(
        data.by_user(),
        [&](const int user_id) -> MatrixXf::RowXpr {
          return user_embedding_.row(user_id);
        },
        item_embedding_, item_gramian_,
        [&](const int user_id) -> float {
          return dual_weight_(user_id);
        });

    float residual_V =
        StepV(data.by_user(), data.by_item(), user_embedding_prev,
              item_embedding_, dual_weight_);

    item_gramian_.noalias() =
        item_embedding_.transpose() * item_embedding_;
    ComputeUserLoss(data.by_user(), user_embedding_, item_embedding_,
                    item_gramian_, user_loss_);
    LOG(INFO) << "Weighted Loss: "
              << (dual_weight_.array() * user_loss_.array()).mean();
    LOG(INFO) << "Mean weights: " << dual_weight_.mean();

    if (print_varstats_) {
      std::vector<float> vals;
      for (int i = 0; i < user_loss_.size(); i++) {
        vals.push_back(-user_loss_[i]);
      }
      auto const Q = vals.size() * alpha_;
      std::nth_element(vals.begin(), vals.begin() + Q, vals.end());
      float loss = 0;
      for (int i = 0; i <= Q; i++) {
        loss += -vals[i];
      }
      LOG(INFO) << "VaR: " << -vals[(int)Q] << " CVaR: " << loss / Q;
      LOG(INFO) << fmt::format(
          "Min: {0:.3f}, Mean: {1:.3f}, Max: {2:.3f}",
          dual_weight_.minCoeff(), dual_weight_.mean(),
          dual_weight_.maxCoeff());
    }
    if (print_residualstats_) {
      LOG(INFO) << fmt::format(
          "U residual: {0}, V residual: {1}, z residual: {2}",
          residual_U, residual_V, residual_z);
    }
    float xi = ComputeXi(user_loss_, prev_xi_);
    LOG(INFO) << "Xi:" << xi;
    prev_xi_ = xi;
  }

  void PrintLosses(const Dataset& data) {
    if (!print_trainstats_) {
      return;
    }
    auto time_start = std::chrono::steady_clock::now();
    // Predict the dataset.
    VectorXf prediction(data.num_tuples());
    for (const auto& user_and_history : data.by_user()) {
      VectorXf user_emb = user_embedding_.row(user_and_history.first);
      for (const auto& item_and_rating_index :
           user_and_history.second) {
        prediction.coeffRef(item_and_rating_index.second) =
            item_embedding_.row(item_and_rating_index.first)
                .dot(user_emb);
      }
    }
    int num_items = item_embedding_.rows();
    int num_users = user_embedding_.rows();

    // Compute observed loss.
    float loss_observed =
        (prediction.array() - 1.0).matrix().squaredNorm();

    // Compute regularizer.
    float loss_reg = 0.0;
    float loss_reg_user_now = 0.0;
    for (auto user_and_history : data.by_user()) {
      loss_reg +=
          user_embedding_.row(user_and_history.first).squaredNorm() *
          UserRegularizationValue(user_and_history.second.size(),
                                  num_items);
      loss_reg_user_now +=
          user_embedding_.row(user_and_history.first).squaredNorm();
    }
    float loss_reg_item_now = 0.0;
    for (auto item_and_history : data.by_item()) {
      loss_reg +=
          item_embedding_.row(item_and_history.first).squaredNorm() *
          ItemRegularizationValue(item_and_history.first, num_users);
      loss_reg_item_now +=
          item_embedding_.row(item_and_history.first).squaredNorm();
    }

    // Unobserved loss.
    MatrixXf user_gramian =
        user_embedding_.transpose() * user_embedding_;
    MatrixXf item_gramian =
        item_embedding_.transpose() * item_embedding_;
    float loss_unobserved =
        (user_gramian.array() * item_gramian.array()).sum();

    float loss = user_loss_.sum();  // + loss_reg_user + loss_reg_item

    VectorXf user_avg_vec = user_embedding_.colwise().mean();
    VectorXf avg_scores = item_embedding_ * user_avg_vec;

    auto time_end = std::chrono::steady_clock::now();

    uint64_t duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            time_end - time_start)
            .count();
    if (std::isnan(loss)) {
      LOG(ERROR) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
      LOG(ERROR) << "NaN is detected!!";
      LOG(ERROR) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
      std::exit(0);
    }
    LOG(INFO) << fmt::format(
        "Loss={0:.2f} Loss_observed={1:.2f} Loss_unobserved={2:.2f} "
        "Loss_reg={3:.2f} Loss_reg (user)={4:.2f} Loss_reg "
        "(item)={5:.2f}",
        loss, loss_observed / data.num_tuples(),
        loss_unobserved / num_items / num_users, loss_reg,
        loss_reg_user_now / num_users, loss_reg_item_now / num_items);
    LOG(INFO) << fmt::format("Time={0}", duration);
  }

  // Computes the regularization value for a user (or item). The value
  // depends on the number of observations for this user (or item) and
  // the total number of items (or users).
  const float UserRegularizationValue(int history_size,
                                      int num_choices) const {
    return regularization_ * (1 + unobserved_weight_ * num_choices);
  }

  const float ItemRegularizationValue(const int item,
                                      int num_choices) const {
    const float num_items = item_embedding_.rows();
    float loss_weights = item_reg_(item);
    return regularization_ *
           (loss_weights + alpha_ * unobserved_weight_ * num_choices);
  }

  template <typename F, typename G>
  inline float StepU(const SpMatrix& data_by_user,
                     F get_user_embedding_ref,
                     const MatrixXf& item_embedding,
                     const MatrixXf& gramian, G get_dual_weight_ref) {
    // Used for per user regularization.
    int num_items = item_embedding.rows();

    std::mutex m;
    auto data_by_user_iter = data_by_user.begin();  // protected by m
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&] {
        while (true) {
          // Get a new user to work on.
          m.lock();
          if (data_by_user_iter == data_by_user.end()) {
            m.unlock();
            return;
          }
          int u = data_by_user_iter->first;
          SpVector train_history = data_by_user_iter->second;
          auto weight = get_dual_weight_ref(u);
          auto user_emb = get_user_embedding_ref(u);
          ++data_by_user_iter;
          m.unlock();

          assert(!train_history.empty());
          float reg = UserRegularizationValue(train_history.size(),
                                              num_items);
          VectorXf new_user_emb = ProjectU(
              train_history, user_emb, item_embedding, gramian, reg,
              unobserved_weight_, weight, stepsize_);
          // Update the user embedding.
          m.lock();
          get_user_embedding_ref(u).noalias() = new_user_emb;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (auto& th : threads) {
      th.join();
    }

    float residual = 0;
    return residual;
  }

  inline float StepV(const SpMatrix& data_by_user,
                     const SpMatrix& data_by_item,
                     const MatrixXf& user_embedding,
                     MatrixXf& item_embedding,
                     const VectorXf& dual_weight) {
    MatrixXf item_embedding_prev;
    if (print_residualstats_) {
      // Copying item embedding matrix for residual computation.
      item_embedding_prev = item_embedding;
    }

    VectorXf norm_dual_weight =
        dual_weight.array() / user_history_size_.array();
    MatrixXf w_user_embedding =
        user_embedding.array() *
        dual_weight.replicate(1, embedding_dim_).array();
    MatrixXf gramian = user_embedding.transpose() * w_user_embedding;

    // Used for per user regularization.
    int num_users = user_embedding.rows();

    std::mutex m;
    auto data_by_item_iter = data_by_item.begin();  // protected by m
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&] {
        while (true) {
          // Get a new item to work on.
          m.lock();
          if (data_by_item_iter == data_by_item.end()) {
            m.unlock();
            return;
          }
          int v = data_by_item_iter->first;
          SpVector train_history = data_by_item_iter->second;
          auto item_emb = item_embedding.row(v);
          ++data_by_item_iter;
          m.unlock();

          assert(!train_history.empty());
          float reg = ItemRegularizationValue(v, num_users);
          VectorXf new_item_emb = ProjectV(
              train_history, item_emb, user_embedding, gramian, reg,
              unobserved_weight_, norm_dual_weight, stepsize_);
          // Update the user embedding.
          m.lock();
          item_embedding.row(v).noalias() = new_item_emb;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (auto& th : threads) {
      th.join();
    }

    float residual = 0;
    if (print_residualstats_) {
      residual = (item_embedding - item_embedding_prev).norm();
    }
    return residual;
  }

  inline void ComputeUserLoss(const SpMatrix& data_by_user,
                              const MatrixXf& user_embedding,
                              const MatrixXf& item_embedding,
                              const MatrixXf& gramian,
                              VectorXf& user_loss) {
    std::mutex m;
    auto data_by_user_iter = data_by_user.begin();  // protected by m
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&] {
        while (true) {
          // Get a new user to work on.
          m.lock();
          if (data_by_user_iter == data_by_user.end()) {
            m.unlock();
            return;
          }
          int u = data_by_user_iter->first;
          VectorXf user_emb = user_embedding.row(u);
          SpVector train_history = data_by_user_iter->second;
          ++data_by_user_iter;
          m.unlock();

          assert(!train_history.empty());
          float loss =
              ComputeLoss(train_history, user_emb, item_embedding,
                          gramian, unobserved_weight_);

          // Update the user embedding.
          m.lock();
          user_loss(u) = loss;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (auto& th : threads) {
      th.join();
    }
  }

  inline float ComputeXi(const VectorXf& user_loss,
                         const float prev_xi) {
    int num_users = user_loss.size();
    std::vector<float> vals;
    vals.reserve(num_users);
    for (int i = 0; i < num_users; i++) {
      vals.push_back(-user_loss[i]);
    }
    auto const Q = vals.size() * alpha_;
    std::nth_element(vals.begin(), vals.begin() + Q, vals.end());
    LOG(INFO) << "Exact Quantile:" << -vals[vals.size() * alpha_];

    return -vals[vals.size() * alpha_];
  }

  inline float ComputeUserWeights(const SpMatrix& data_by_user,
                                  VectorXf& dual_weight,
                                  const VectorXf& user_loss,
                                  const float xi) {
    VectorXf dual_weight_prev;
    if (print_residualstats_) {
      dual_weight_prev = dual_weight;
    }
    std::mutex m;
    auto data_by_user_iter = data_by_user.begin();  // protected by m
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&] {
        while (true) {
          // Get a new user to work on.
          m.lock();
          if (data_by_user_iter == data_by_user.end()) {
            m.unlock();
            return;
          }
          int u = data_by_user_iter->first;
          float loss = user_loss(u);
          ++data_by_user_iter;
          m.unlock();

          float new_weight = (loss - xi) >= 0;

          // Update the user embedding.
          m.lock();
          dual_weight(u) = new_weight;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (auto& th : threads) {
      th.join();
    }

    float residual = 0;
    if (print_residualstats_) {
      residual = (dual_weight - dual_weight_prev).norm();
    }
    return residual;
  }

  template <typename F, typename G>
  inline float StepU_eval(const SpMatrix& data_by_user,
                          F get_user_embedding_ref,
                          const MatrixXf& item_embedding,
                          const MatrixXf& gramian,
                          G get_dual_weight_ref) {
    // Used for per user regularization.
    int num_items = item_embedding.rows();

    std::mutex m;
    auto data_by_user_iter = data_by_user.begin();  // protected by m
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&] {
        while (true) {
          // Get a new user to work on.
          m.lock();
          if (data_by_user_iter == data_by_user.end()) {
            m.unlock();
            return;
          }
          int u = data_by_user_iter->first;
          SpVector train_history = data_by_user_iter->second;
          auto weight = get_dual_weight_ref(u);
          ++data_by_user_iter;
          m.unlock();

          assert(!train_history.empty());
          float reg = UserRegularizationValue(train_history.size(),
                                              num_items);
          VectorXf new_user_emb =
              ProjectU_eval(train_history, item_embedding, gramian,
                            reg, unobserved_weight_, weight);
          // Update the user embedding.
          m.lock();
          get_user_embedding_ref(u).noalias() = new_user_emb;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (auto& th : threads) {
      th.join();
    }

    float residual = 0;
    return residual;
  }

  const MatrixXf& item_embedding() const {
    return item_embedding_;
  }

  void SetPrintTrainStats(const bool print_trainstats) {
    print_trainstats_ = print_trainstats;
  }

  void SetPrintResidualStats(const bool print_residualstats) {
    print_residualstats_ = print_residualstats;
  }

  void SetPrintVarStats(const bool print_varstats) {
    print_varstats_ = print_varstats;
  }

  void Initialize(const Dataset& data) {
    ComputeUserLoss(data.by_user(), user_embedding_, item_embedding_,
                    item_gramian_, user_loss_);
    float prev_xi = user_loss_.mean();

    for (auto& user_and_history : data.by_user()) {
      user_history_size_(user_and_history.first) =
          user_and_history.second.size();
    }
    for (auto& item_and_history : data.by_item()) {
      const int v = item_and_history.first;
      for (auto user_and_rating_index : item_and_history.second) {
        const int u = user_and_rating_index.first;
        item_reg_(v) += 1.0 / user_history_size_(u);
      }
    }
  }

private:
  MatrixXf user_embedding_;
  MatrixXf item_embedding_;
  MatrixXf item_gramian_;
  VectorXf user_loss_;
  VectorXf dual_weight_;
  VectorXf user_history_size_;
  VectorXf item_reg_;

  float regularization_;
  int embedding_dim_;
  float unobserved_weight_;
  float alpha_;
  float prev_xi_;
  float residual_tolerance_;
  float stepsize_;

  bool print_trainstats_;
  bool print_residualstats_;
  bool print_varstats_;
};

}  // namespace frecsys
