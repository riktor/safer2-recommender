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
// permissions and limitations under the License.
//
// Branched from
// https://github.com/google-research/google-research/tree/master/ials
// with modification.
#pragma once

#include <algorithm>
#include <chrono>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include <Eigen/IterativeLinearSolvers>
#include "Eigen/Core"
#include "Eigen/Dense"

#include "frecsys/recommender.h"

namespace frecsys {

class IALSRecommender : public Recommender {
public:
  IALSRecommender(int embedding_dim, int num_users, int num_items,
                  float reg, float reg_exp, float unobserved_weight,
                  float stdev, float alpha, bool use_cg,
                  float cg_error_tolerance, int cg_max_iterations)
      : user_embedding_(num_users, embedding_dim),
        item_embedding_(num_items, embedding_dim),
        user_loss_(num_users) {
    // Initialize embedding matrices
    float adjusted_stdev = stdev / sqrt(embedding_dim);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    init_matrix(&user_embedding_, gen, adjusted_stdev);
    init_matrix(&item_embedding_, gen, adjusted_stdev);

    regularization_ = reg;
    regularization_exp_ = reg_exp;
    embedding_dim_ = embedding_dim;
    unobserved_weight_ = unobserved_weight;

    alpha_ = alpha;

    use_cg_ = use_cg;
    cg_error_tolerance_ = cg_error_tolerance;
    cg_max_iterations_ = cg_max_iterations;
  }

  VectorXf Score(const int user_id,
                 const SpVector& user_history) override {
    throw("Function 'Score' is not implemented");
  }

  static const float ComputeLoss(const SpVector& user_history,
                                 const VectorXf& user_emb,
                                 const MatrixXf& item_embeddings,
                                 const MatrixXf& item_gramian,
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
    return loss;
  }

  inline static const VectorXf Project(
      const SpVector& user_history, const MatrixXf& item_embeddings,
      const MatrixXf& gramian, const float reg,
      const float unobserved_weight, bool use_cg,
      const float cg_error_tolerance = 1e-10,
      const int cg_max_iterations = 100) {
    assert(user_history.size() > 0);

    int embedding_dim = item_embeddings.cols();
    assert(embedding_dim > 0);

    VectorXf new_value(embedding_dim);

    MatrixXf matrix = unobserved_weight * gramian;

    for (int i = 0; i < embedding_dim; ++i) {
      matrix(i, i) += reg;
    }

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
      factor_batch.col(num_batched).noalias() = cp_v;
      rhs.noalias() += cp_v;

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

    if (use_cg) {
      Eigen::ConjugateGradient<MatrixXf, Eigen::Lower> cg;
      cg.compute(matrix);
      cg.setMaxIterations(cg_max_iterations);
      cg.setTolerance(cg_error_tolerance);
      return cg.solve(rhs);
    } else {
      Eigen::LLT<MatrixXf, Eigen::Lower> cholesky(matrix);
      assert(cholesky.info() == Eigen::Success);
      return cholesky.solve(rhs);
    }
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
    Step(
        data.by_user(),
        [&](const int user_id) -> MatrixXf::RowXpr {
          return user_embedding.row(user_to_ind[user_id]);
        },
        item_embedding_);

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
    float residual_U = Step(
        data.by_user(),
        [&](const int index) -> MatrixXf::RowXpr {
          return user_embedding_.row(index);
        },
        item_embedding_);

    // Optimize the item embeddings
    float residual_V = Step(
        data.by_item(),
        [&](const int index) -> MatrixXf::RowXpr {
          return item_embedding_.row(index);
        },
        user_embedding_);

    ComputeLosses(data);

    ComputeUserLoss(data.by_user(), user_embedding_, item_embedding_,
                    user_loss_);
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
    }
    if (print_residualstats_) {
      LOG(INFO) << fmt::format("U residual: {0}, V residual: {1}",
                               residual_U, residual_V);
    }
  }

  void ComputeLosses(const Dataset& data) {
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
    double loss_reg = 0.0;
    double loss_reg_user_now = 0.0;
    for (auto user_and_history : data.by_user()) {
      loss_reg +=
          user_embedding_.row(user_and_history.first).squaredNorm() *
          RegularizationValue(user_and_history.second.size(),
                              num_items);
      loss_reg_user_now +=
          user_embedding_.row(user_and_history.first).squaredNorm();
    }
    double loss_reg_item_now = 0.0;
    for (auto item_and_history : data.by_item()) {
      loss_reg +=
          item_embedding_.row(item_and_history.first).squaredNorm() *
          RegularizationValue(item_and_history.second.size(),
                              num_users);
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

    float loss = loss_observed +
                 this->unobserved_weight_ * loss_unobserved +
                 loss_reg;

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
  const float RegularizationValue(int history_size,
                                  int num_choices) const {
    return this->regularization_ *
           pow(history_size + this->unobserved_weight_ * num_choices,
               this->regularization_exp_);
  }

  template <typename F>
  inline float Step(const SpMatrix& data_by_user,
                    F get_user_embedding_ref,
                    const MatrixXf& item_embedding) {
    MatrixXf gramian = item_embedding.transpose() * item_embedding;

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
          ++data_by_user_iter;
          m.unlock();

          assert(!train_history.empty());
          float reg =
              RegularizationValue(train_history.size(), num_items);
          VectorXf new_user_emb =
              Project(train_history, item_embedding, gramian, reg,
                      this->unobserved_weight_, use_cg_,
                      cg_error_tolerance_, cg_max_iterations_);
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

  inline void ComputeUserLoss(const SpMatrix& data_by_user,
                              const MatrixXf& user_embedding,
                              const MatrixXf& item_embedding,
                              VectorXf& user_loss) {
    MatrixXf gramian = item_embedding.transpose() * item_embedding;

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

  const MatrixXf& item_embedding() const {
    return item_embedding_;
  }

  void SetPrintTrainStats(const bool print_trainstats) override {
    print_trainstats_ = print_trainstats;
  }

  void SetPrintResidualStats(
      const bool print_residualstats) override {
    print_residualstats_ = print_residualstats;
  }

  void SetPrintVarStats(const bool print_varstats) override {
    print_varstats_ = print_varstats;
  }

private:
  MatrixXf user_embedding_;
  MatrixXf item_embedding_;
  VectorXf user_loss_;

  float regularization_;
  float regularization_exp_;
  int embedding_dim_;
  float unobserved_weight_;
  float alpha_;

  bool use_cg_;
  float cg_error_tolerance_;
  int cg_max_iterations_;

  bool print_trainstats_;
  bool print_residualstats_;
  bool print_varstats_;
};
}  // namespace frecsys
