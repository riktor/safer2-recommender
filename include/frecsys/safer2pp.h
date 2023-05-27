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

class SAFER2ppRecommender : public Recommender {
public:
  SAFER2ppRecommender(int embedding_dim, int num_users, int num_items,
                      float reg, float unobserved_weight,
                      float bandwidth, float alpha, float stdev,
                      int xi_iterations, int pd_iterations,
                      bool use_epanechnikov, bool use_snr,
                      float sampling_ratio, int block_size)
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
    bandwidth_ = bandwidth;
    alpha_ = alpha;
    prev_xi_ = 0.0;
    use_epanechnikov_ = use_epanechnikov;

    xi_iterations_ = xi_iterations;
    use_snr_ = use_snr;
    sampling_ratio_ = sampling_ratio;

    pd_iterations_ = pd_iterations;
    block_size_ = block_size;
  }

  VectorXf Score(const int user_id,
                 const SpVector& user_history) override {
    throw("Function 'Score' is not implemented");
  }

  inline static const float ComputeLoss(
      const SpVector& user_history, const VectorXf& user_emb,
      const MatrixXf& item_embeddings, const VectorXf& prediction,
      const MatrixXf& item_gramian, const float beta) {
    int num_items = item_embeddings.rows();
    float loss = 0;
    for (const auto& item_and_rating_index : user_history) {
      const int rating_index = item_and_rating_index.second;
      loss += pow(prediction.coeff(rating_index) - 1, 2.0);
    }
    loss /= user_history.size();
    float ireg = user_emb.transpose() * item_gramian * user_emb;
    loss += beta * ireg;
    loss /= 2.0;
    return loss;
  }

  inline static const VectorXf ProjectU(
      const SpVector& user_history, const VectorXf& user_embedding,
      const VectorXf& local_user_embedding,
      const MatrixXf& local_item_embedding,
      const VectorXf& prediction, const MatrixXf& local_gramian,
      const MatrixXf& local_global_gramian, const float reg,
      const float unobserved_weight, const float weight) {
    int history_size = user_history.size();
    int local_embedding_dim = local_item_embedding.cols();
    assert(history_size > 0);
    assert(local_embedding_dim > 0);

    MatrixXf matrix =
        MatrixXf::Zero(local_embedding_dim, local_embedding_dim);
    const int kMaxBatchSize = 128;
    auto matrix_symm = matrix.selfadjointView<Eigen::Lower>();
    VectorXf rhs = VectorXf::Zero(local_embedding_dim);
    const int batch_size = std::min(
        static_cast<int>(user_history.size()), kMaxBatchSize);
    int num_batched = 0;
    MatrixXf factor_batch(local_embedding_dim, batch_size);
    for (const auto& item_and_rating_index : user_history) {
      const int cp = item_and_rating_index.first;
      const int rating_index = item_and_rating_index.second;
      assert(cp < local_item_embedding.rows());
      assert(rating_index < prediction.size());
      const VectorXf cp_v = local_item_embedding.row(cp);

      const float residual = (prediction.coeff(rating_index) - 1.0);

      factor_batch.col(num_batched).noalias() = cp_v;
      rhs.noalias() += cp_v * residual;

      ++num_batched;
      if (num_batched == batch_size) {
        matrix_symm.rankUpdate(factor_batch);
        num_batched = 0;
      }
    }
    if (num_batched != 0) {
      auto factor_block =
          factor_batch.block(0, 0, local_embedding_dim, num_batched);
      matrix_symm.rankUpdate(factor_block);
    }
    matrix /= history_size;
    matrix.noalias() += unobserved_weight * local_gramian;
    matrix *= weight;
    rhs *= weight / history_size;

    // add "prediction" for the unobserved items
    rhs += unobserved_weight * local_global_gramian * user_embedding *
           weight;
    // add the regularization.
    rhs += reg * local_user_embedding;

    for (int i = 0; i < local_embedding_dim; ++i) {
      matrix(i, i) += reg;
    }

    Eigen::LLT<Eigen::MatrixXf, Eigen::Lower> cholesky(matrix);
    assert(cholesky.info() == Eigen::Success);
    return local_user_embedding - cholesky.solve(rhs);
  }

  inline static const VectorXf ProjectV(
      const SpVector& item_history, const VectorXf& item_embedding,
      const VectorXf& local_item_embedding,
      const MatrixXf& local_user_embedding,
      const VectorXf& prediction, const MatrixXf& local_gramian,
      const MatrixXf& local_global_gramian, const float reg,
      const float unobserved_weight, const VectorXf& dual_weight) {
    int history_size = item_history.size();
    int local_embedding_dim = local_user_embedding.cols();
    assert(history_size > 0);
    assert(local_embedding_dim > 0);

    MatrixXf matrix = unobserved_weight * local_gramian;

    for (int i = 0; i < local_embedding_dim; ++i) {
      matrix(i, i) += reg;
    }

    const int kMaxBatchSize = 128;
    auto matrix_symm = matrix.selfadjointView<Eigen::Lower>();
    VectorXf rhs = VectorXf::Zero(local_embedding_dim);
    const int batch_size = std::min(
        static_cast<int>(item_history.size()), kMaxBatchSize);
    int num_batched = 0;
    MatrixXf factor_batch(local_embedding_dim, batch_size);
    for (const auto& user_and_rating_index : item_history) {
      const int cp = user_and_rating_index.first;
      const int rating_index = user_and_rating_index.second;
      const VectorXf cp_v = local_user_embedding.row(cp);
      const float w = dual_weight(cp);

      const float residual = (prediction.coeff(rating_index) - 1.0);

      factor_batch.col(num_batched).noalias() = cp_v * sqrt(w);
      rhs.noalias() += cp_v * residual * w;

      ++num_batched;
      if (num_batched == batch_size) {
        matrix_symm.rankUpdate(factor_batch);
        num_batched = 0;
      }
    }
    if (num_batched != 0) {
      auto factor_block =
          factor_batch.block(0, 0, local_embedding_dim, num_batched);
      matrix_symm.rankUpdate(factor_block);
    }

    // add "prediction" for the unobserved items
    rhs += unobserved_weight * local_global_gramian * item_embedding;
    // add the regularization.
    rhs += reg * local_item_embedding;
    Eigen::LLT<Eigen::MatrixXf, Eigen::Lower> cholesky(matrix);
    assert(cholesky.info() == Eigen::Success);
    return local_item_embedding - cholesky.solve(rhs);
  }

  // Custom implementation of EvaluateDataset that does the projection
  // using the iterative optimization algorithm.
  EvaluationResult EvaluateDataset(
      const VectorXi& k_list, const VectorXf& alpha_list,
      const Dataset& data, const SpMatrix& eval_by_user) override {
    int num_epochs = 8;

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
    // Train the user embeddings for num_epochs.
    for (int e = 0; e < num_epochs; ++e) {
      // Predict the dataset using the new user embeddings and the
      // existing item embeddings.
      PredictDataset(
          data.by_user(), &prediction,
          [&](const int user_id) -> MatrixXf::RowXpr {
            return user_embedding.row(user_to_ind[user_id]);
          },
          item_embedding_);
      for (const auto& user_and_history : data.by_user()) {
        const VectorXf& user_emb =
            user_embedding.row(user_to_ind[user_and_history.first]);
        for (const auto& item_and_rating_index :
             user_and_history.second) {
          prediction.coeffRef(item_and_rating_index.second) =
              item_embedding_.row(item_and_rating_index.first)
                  .dot(user_emb);
        }
      }
      // Optimize the user embeddings for each block.
      for (int start = 0; start < embedding_dim_;
           start += block_size_) {
        assert(start < embedding_dim_);
        int end = std::min(start + block_size_, embedding_dim_);

        StepU(
            data.by_user(), start, end, &prediction,
            [&](const int user_id) -> MatrixXf::RowXpr {
              return user_embedding.row(user_to_ind[user_id]);
            },
            item_embedding_,
            [&](const int user_id) -> float { return 1.0; });
      }
    }

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
    // Predict the dataset.
    VectorXf prediction(data.num_tuples());
    PredictDataset(
        data.by_user(), &prediction,
        [&](const int user_id) -> MatrixXf::RowXpr {
          return user_embedding_.row(user_id);
        },
        item_embedding_);
    for (int t = 0; t < pd_iterations_; ++t) {
      float residual_U = 0;
      float residual_V = 0;
      float residual_z = ComputeUserWeights(
          data.by_user(), dual_weight_, user_loss_, prev_xi_);
      for (int start = 0; start < embedding_dim_;
           start += block_size_) {
        assert(start < embedding_dim_);
        int end = std::min(start + block_size_, embedding_dim_);
        residual_U += StepU(
            data.by_user(), start, end, &prediction,
            [&](const int user_id) -> MatrixXf::RowXpr {
              return user_embedding_.row(user_id);
            },
            item_embedding_,
            [&](const int user_id) -> float {
              return dual_weight_(user_id);
            });
        residual_V += StepV(data.by_user(), data.by_item(), start,
                            end, &prediction, item_embedding_,
                            user_embedding_, dual_weight_);
      }
      item_gramian_.noalias() =
          item_embedding_.transpose() * item_embedding_;
      ComputeUserLoss(data.by_user(), user_embedding_,
                      item_embedding_, prediction, item_gramian_,
                      user_loss_);
      LOG(INFO) << "Weighted Loss: "
                << (dual_weight_.array() * user_loss_.array()).mean();

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
        LOG(INFO) << "VaR: " << -vals[(int)Q]
                  << " CVaR: " << loss / Q;
        LOG(INFO) << fmt::format(
            "Min: {0:.3f}, Mean: {1:.3f}, Max: {2:.3f}",
            dual_weight_.minCoeff(), dual_weight_.mean(),
            dual_weight_.maxCoeff());
      }
      if (print_residualstats_) {
        LOG(INFO) << fmt::format(
            "U residual: {0}, V residual: {1}, z residual: {2}",
            sqrt(residual_U), sqrt(residual_V), residual_z);
      }
    }
    float xi = ComputeXi(user_loss_, prev_xi_, xi_iterations_);
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
    PredictDataset(
        data.by_user(), &prediction,
        [&](const int user_id) -> MatrixXf::RowXpr {
          return user_embedding_.row(user_id);
        },
        item_embedding_);
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
    float num_items = item_embedding_.rows();
    float loss_weights = item_reg_(item);
    return regularization_ *
           (loss_weights + alpha_ * unobserved_weight_ * num_choices);
  }

  template <typename F, typename G>
  inline float StepU(const SpMatrix& data_by_user,
                     const int block_start, const int block_end,
                     VectorXf* prediction, F get_user_embedding_ref,
                     const MatrixXf& item_embedding,
                     G get_dual_weight_ref) {
    MatrixXf local_item_emb =
        item_embedding.block(0, block_start, item_embedding.rows(),
                             block_end - block_start);

    MatrixXf local_gramian =
        local_item_emb.transpose() * local_item_emb;
    MatrixXf local_global_gramian =
        local_item_emb.transpose() * item_embedding;

    // Used for per user regularization.
    int num_items = item_embedding.rows();

    std::mutex m;
    float residual = 0;
    auto data_by_user_iter = data_by_user.begin();
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
          ++data_by_user_iter;
          m.unlock();

          assert(!train_history.empty());
          float reg = UserRegularizationValue(train_history.size(),
                                              num_items);
          VectorXf old_user_emb = get_user_embedding_ref(u);
          VectorXf old_local_user_emb = old_user_emb.segment(
              block_start, block_end - block_start);
          float weight = get_dual_weight_ref(u);

          VectorXf new_local_user_emb = ProjectU(
              train_history, old_user_emb, old_local_user_emb,
              local_item_emb, *prediction, local_gramian,
              local_global_gramian, reg, unobserved_weight_, weight);

          VectorXf delta_local_user_emb =
              new_local_user_emb - old_local_user_emb;

          if (print_residualstats_) {
            residual += delta_local_user_emb.squaredNorm();
          }
          for (const auto& item_and_rating_index : train_history) {
            prediction->coeffRef(item_and_rating_index.second) +=
                delta_local_user_emb.dot(
                    local_item_emb.row(item_and_rating_index.first));
          }
          // Update the user embedding.
          m.lock();
          get_user_embedding_ref(u)
              .segment(block_start, block_end - block_start)
              .noalias() = new_local_user_emb;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (auto& th : threads) {
      th.join();
    }

    return residual;
  }

  inline float StepV(const SpMatrix& data_by_user,
                     const SpMatrix& data_by_item,
                     const int block_start, const int block_end,
                     VectorXf* prediction, MatrixXf& item_embedding,
                     const MatrixXf& user_embedding,
                     const VectorXf& dual_weight) {
    VectorXf norm_dual_weight =
        dual_weight.array() / user_history_size_.array();
    MatrixXf local_user_emb =
        user_embedding.block(0, block_start, user_embedding.rows(),
                             block_end - block_start);
    int local_embedding_dim = local_user_emb.cols();
    MatrixXf local_user_emb_w =
        local_user_emb.array() *
        dual_weight.replicate(1, local_embedding_dim).array();
    MatrixXf local_gramian =
        local_user_emb_w.transpose() * local_user_emb;
    MatrixXf local_global_gramian =
        local_user_emb_w.transpose() * user_embedding;

    MatrixXf item_embedding_prev;
    if (print_residualstats_) {
      // Copying item embedding matrix for residual computation.
      item_embedding_prev = item_embedding;
    }

    // Used for per user regularization.
    int num_users = user_embedding.rows();

    std::mutex m;
    auto data_by_item_iter = data_by_item.begin();  // protected by m
    float residual = 0;
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
          ++data_by_item_iter;
          m.unlock();

          assert(!train_history.empty());
          float reg = ItemRegularizationValue(v, num_users);
          VectorXf old_item_emb = item_embedding.row(v);
          VectorXf old_local_item_emb = old_item_emb.segment(
              block_start, block_end - block_start);
          VectorXf new_local_item_emb = ProjectV(
              train_history, old_item_emb, old_local_item_emb,
              local_user_emb, *prediction, local_gramian,
              local_global_gramian, reg, unobserved_weight_,
              norm_dual_weight);
          // Update the ratings (without a lock)
          VectorXf delta_local_item_emb =
              new_local_item_emb - old_local_item_emb;
          for (const auto& user_and_rating_index : train_history) {
            prediction->coeffRef(user_and_rating_index.second) +=
                delta_local_item_emb.dot(
                    local_user_emb.row(user_and_rating_index.first));
          }
          if (print_residualstats_) {
            residual += delta_local_item_emb.squaredNorm();
          }
          // Update the user embedding.
          m.lock();
          item_embedding.row(v).segment(block_start,
                                        block_end - block_start) =
              new_local_item_emb;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (auto& th : threads) {
      th.join();
    }
    return residual;
  }

  inline void ComputeUserLoss(const SpMatrix& data_by_user,
                              const MatrixXf& user_embedding,
                              const MatrixXf& item_embedding,
                              const VectorXf& prediction,
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
                          prediction, gramian, unobserved_weight_);

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

  template <typename F>
  inline void PredictDataset(const SpMatrix& data_by_user,
                             VectorXf* prediction,
                             F get_user_embedding_ref,
                             const MatrixXf& item_embedding) {
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
          ++data_by_user_iter;
          m.unlock();

          assert(!train_history.empty());
          VectorXf user_emb = get_user_embedding_ref(u);
          VectorXf pred(train_history.size());
          VectorXi indices(train_history.size());
          int c = 0;
          for (const auto& item_and_rating_index : train_history) {
            pred[c] = item_embedding_.row(item_and_rating_index.first)
                          .dot(user_emb);
            indices[c] = item_and_rating_index.second;
            c++;
          }

          // Update the user embedding.
          m.lock();
          for (int i = 0; i < c; i++) {
            prediction->coeffRef(indices[i]) = pred[i];
          }
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (auto& th : threads) {
      th.join();
    }
  }

  // Helper functions for Gaussian kernel
  inline static float gaussian_kernel(const float u, const float h) {
    return pow(2 * M_PI, -0.5) * exp(-pow((u / h) * M_SQRT1_2, 2)) /
           h;
  }

  inline static float gaussian_kernel_cdf(const float u,
                                          const float h) {
    return 0.5 * erfc(-(u / h) * M_SQRT1_2);
  }

  inline static float gaussian_loss(const float u, const float h,
                                    const float alpha) {
    // Integral of Gaussian CDF (Remark 3.1 of He et al. 2021)
    float ell = h * gaussian_kernel(u, h) +
                (u / h) * (1 - 2 * gaussian_kernel_cdf(-u, h));
    return (h / 2) * ell + ((1 - alpha) - 0.5) * u;
  }

  // Helper functions for Epanechnikov kernel
  inline static float epanechnikov_kernel(const float u,
                                          const float h) {
    float uh = u / h;
    return (3.0 / 4.0) * (1 - pow(uh, 2)) * (int)(abs(uh) < 1) / h;
  }

  inline static float epanechnikov_kernel_cdf(const float u,
                                              const float h) {
    float uh = u / h;
    int in_supp = (int)(abs(uh) <= 1);
    int pos = (int)(uh > 1);
    float cdf =
        (pow(h, -3) / 4.0) *
            ((3 * u * pow(h, 2) - pow(u, 3)) + 2 * pow(h, 3)) *
            in_supp +
        (1 - in_supp) * pos;
    return cdf;
  }

  inline static float epanechnikov_loss(const float u, const float h,
                                        const float alpha) {
    // Integral of Epanechnikov CDF (Remark 3.1 of He et al. 2021)
    float uh = u / h;
    int in_supp = (int)(abs(uh) <= 1);
    int pos = (int)(uh > 1);
    float ell = ((3.0 / 4.0) * pow(uh, 2) - (1.0 / 8.0) * pow(uh, 4) +
                 (3.0 / 8.0)) *
                    in_supp +
                abs(uh) * pos;
    return (1.0 / 2.0) * h * ell + ((1 - alpha) - 0.5) * u;
  }

  inline std::tuple<float, float, float> EvaluateQuantile(
      const float xi, const VectorXf& user_loss) {
    const int n_users = user_loss.size();

    const auto r = user_loss.array() - xi;

    float grad, H, value;
    if (use_epanechnikov_) {
      grad = -(1 - alpha_) +
             r.unaryExpr([&](float u) {
                return epanechnikov_kernel_cdf(-u, bandwidth_);
              }).mean();
      H = r.unaryExpr([&](float u) {
             return epanechnikov_kernel(-u, bandwidth_);
           }).mean();
      value = r.unaryExpr([&](float u) {
                 return epanechnikov_loss(u, bandwidth_, alpha_);
               }).mean();
    } else {
      grad = -(1 - alpha_) +
             r.unaryExpr([&](float u) {
                return gaussian_kernel_cdf(-u, bandwidth_);
              }).mean();
      H = r.unaryExpr([&](float u) {
             return gaussian_kernel(-u, bandwidth_);
           }).mean();
      value = r.unaryExpr([&](float u) {
                 return gaussian_loss(u, bandwidth_, alpha_);
               }).mean();
    }
    return {value / alpha_, grad / alpha_, H / alpha_};
  }

  inline float ComputeXiDirection(float xi,
                                  const VectorXf& user_loss) {
    auto [f0, grad_f0, H] = EvaluateQuantile(xi, user_loss);

    const float d = grad_f0 / H;

    // Armijo linesearch
    const float c = 1e-4;
    float gamma = 1.0;
    float x = xi + gamma * (-d);
    for (int k = 0; k < 32; k++) {
      auto [fx, grad_fx, H_fx] = EvaluateQuantile(x, user_loss);
      if (fx > f0 + c * gamma * grad_fx * (-d)) {
        gamma *= 0.5;
        x = xi + gamma * (-d);
      } else {
        break;
      }
    }
    return -gamma * d;
  }

  inline float ComputeXi(const VectorXf& user_loss,
                         const float prev_xi,
                         const int nr_iterations) {
    int num_users = user_loss.size();

    float xi = prev_xi;
    for (int t = 0; t < nr_iterations; ++t) {
      float d = 0;
      if (!use_snr_) {
        d = ComputeXiDirection(xi, user_loss);
      } else {
        // Sub-sampled Newton-Raphson Method
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> uni(0, num_users - 1);
        std::vector<int> sample_inds;
        int num_samples = num_users * sampling_ratio_;
        for (int j = 0; j < num_samples; j++) {
          auto i = uni(rng);
          sample_inds.push_back(i);
        }
        d = ComputeXiDirection(xi, user_loss(sample_inds));
      }
      xi = xi + d;
    }
    return xi;
  }

  inline float ComputeUserWeights(const SpMatrix& data_by_user,
                                  VectorXf& dual_weight,
                                  const VectorXf& user_loss,
                                  const float xi) {
    VectorXf dual_weight_prev;
    if (print_residualstats_) {
      dual_weight_prev = dual_weight;
    }
    VectorXf rs = user_loss.array() - xi;
    if (use_epanechnikov_) {
      dual_weight.noalias() = rs.unaryExpr([&](float u) {
        return 1 - epanechnikov_kernel_cdf(-u, bandwidth_);
      });
    } else {
      dual_weight.noalias() = rs.unaryExpr([&](float u) {
        return 1 - gaussian_kernel_cdf(-u, bandwidth_);
      });
    }
    float residual = 0;
    if (print_residualstats_) {
      residual = (dual_weight - dual_weight_prev).norm();
    }
    return residual;
  }

  const MatrixXf& item_embedding() const {
    return item_embedding_;
  }

  void SetPrintTrainStats(const bool print_trainstats) {
    print_trainstats_ = print_trainstats;
  }

  void SetPrintResidualStats(
      const bool print_residualstats) override {
    print_residualstats_ = print_residualstats;
  }

  void SetPrintVarStats(const bool print_varstats) override {
    print_varstats_ = print_varstats;
  }

  float GetMeanWeight() const {
    return dual_weight_.mean();
  }

  void Initialize(const Dataset& data) {
    VectorXf prediction(data.num_tuples());
    PredictDataset(
        data.by_user(), &prediction,
        [&](const int user_id) -> MatrixXf::RowXpr {
          return user_embedding_.row(user_id);
        },
        item_embedding_);

    ComputeUserLoss(data.by_user(), user_embedding_, item_embedding_,
                    prediction, item_gramian_, user_loss_);
    float prev_xi = user_loss_.mean();
    float xi = ComputeXi(user_loss_, prev_xi, xi_iterations_);
    LOG(INFO) << "Initial Xi:" << xi;
    prev_xi_ = xi;

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
  float regularization_exp_;
  int embedding_dim_;
  float unobserved_weight_;
  float bandwidth_;
  float alpha_;
  float prev_xi_;
  bool use_epanechnikov_;

  int xi_iterations_;
  bool use_snr_;
  float sampling_ratio_;

  int pd_iterations_;
  int block_size_;

  bool print_trainstats_;
  bool print_residualstats_;
  bool print_varstats_;
};

}  // namespace frecsys
