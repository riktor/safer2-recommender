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

class IALSppRecommender : public Recommender {
public:
  IALSppRecommender(int embedding_dim, int num_users, int num_items,
                    float reg, float reg_exp, float unobserved_weight,
                    float stdev, float alpha, int block_size)
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

    block_size_ = block_size;
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

  inline const VectorXf ProjectBlock(
      const SpVector& user_history, const VectorXf& user_embedding,
      const VectorXf& local_user_embedding,
      const MatrixXf& local_item_embedding,
      const VectorXf& prediction, const MatrixXf& local_gramian,
      const MatrixXf& local_global_gramian, const float reg,
      const float unobserved_weight) {
    assert(user_history.size() > 0);
    int local_embedding_dim = local_item_embedding.cols();
    assert(local_embedding_dim > 0);

    VectorXf new_value(local_embedding_dim);

    MatrixXf matrix = unobserved_weight * local_gramian;

    for (int i = 0; i < local_embedding_dim; ++i) {
      matrix(i, i) += reg;
    }

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

      factor_batch.col(num_batched) = cp_v;
      rhs += cp_v * residual;

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
    rhs += unobserved_weight * local_global_gramian * user_embedding;
    // add the regularization.
    rhs += reg * local_user_embedding;

    Eigen::LLT<MatrixXf, Eigen::Lower> cholesky(matrix);
    assert(cholesky.info() == Eigen::Success);
    new_value = local_user_embedding - cholesky.solve(rhs);

    return new_value;
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

      // Optimize the user embeddings for each block.
      for (int start = 0; start < embedding_dim_;
           start += block_size_) {
        assert(start < embedding_dim_);
        int end = std::min(start + block_size_, embedding_dim_);

        Step(
            data.by_user(), start, end, &prediction,
            [&](const int user_id) -> MatrixXf::RowXpr {
              return user_embedding.row(user_to_ind[user_id]);
            },
            item_embedding_);
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
    // Predict the dataset.
    VectorXf prediction(data.num_tuples());
    PredictDataset(
        data.by_user(), &prediction,
        [&](const int user_id) -> MatrixXf::RowXpr {
          return user_embedding_.row(user_id);
        },
        item_embedding_);

    float residual_U = 0;
    float residual_V = 0;
    for (int start = 0; start < embedding_dim_;
         start += block_size_) {
      assert(start < embedding_dim_);
      int end = std::min(start + block_size_, embedding_dim_);

      residual_U += Step(
          data.by_user(), start, end, &prediction,
          [&](const int index) -> MatrixXf::RowXpr {
            return user_embedding_.row(index);
          },
          item_embedding_);

      // Optimize the item embeddings
      residual_V += Step(
          data.by_item(), start, end, &prediction,
          [&](const int index) -> MatrixXf::RowXpr {
            return item_embedding_.row(index);
          },
          user_embedding_);
    }

    ComputeLosses(data);
    if (print_varstats_) {
      ComputeUserLoss(data.by_user(), user_embedding_,
                      item_embedding_, user_loss_);
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
                               sqrt(residual_U), sqrt(residual_V));
    }
  }

  void ComputeLosses(const Dataset& data) {
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
                    const int block_start, const int block_end,
                    VectorXf* prediction, F get_user_embedding_ref,
                    const MatrixXf& item_embedding) {
    MatrixXf local_item_emb =
        item_embedding.block(0, block_start, item_embedding.rows(),
                             block_end - block_start);

    // TODO: consider creating the local_gramian as a block from the
    // local_global_gramian
    MatrixXf local_gramian =
        local_item_emb.transpose() * local_item_emb;
    MatrixXf local_global_gramian =
        local_item_emb.transpose() * item_embedding;

    // Used for per user regularization.
    int num_items = item_embedding.rows();

    std::mutex m;
    float residual = 0;
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
          VectorXf old_user_emb = get_user_embedding_ref(u);
          VectorXf old_local_user_emb = old_user_emb.segment(
              block_start, block_end - block_start);
          VectorXf new_local_user_emb = ProjectBlock(
              train_history, old_user_emb, old_local_user_emb,
              local_item_emb, *prediction, local_gramian,
              local_global_gramian, reg, this->unobserved_weight_);
          // Update the ratings (without a lock)
          VectorXf delta_local_user_emb =
              new_local_user_emb - old_local_user_emb;
          for (const auto& item_and_rating_index : train_history) {
            prediction->coeffRef(item_and_rating_index.second) +=
                delta_local_user_emb.dot(
                    local_item_emb.row(item_and_rating_index.first));
          }
          if (print_residualstats_) {
            residual += delta_local_user_emb.squaredNorm();
          }
          // Update the user embedding.
          m.lock();
          get_user_embedding_ref(u).segment(block_start,
                                            block_end - block_start) =
              new_local_user_emb;
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

  int block_size_;

  bool print_trainstats_;
  bool print_residualstats_;
  bool print_varstats_;
};
}  // namespace frecsys
