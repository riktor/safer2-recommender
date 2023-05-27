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
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "frecsys/dataset.h"
#include "frecsys/evaluation.h"
#include "frecsys/types.h"

namespace frecsys {

class Recommender {
public:
  virtual ~Recommender() {}

  virtual VectorXf Score(const int user_id,
                         const SpVector& user_history) {
    return VectorXf::Zero(1);
  }

  // Common implementation for evaluating a dataset. It uses the
  // scoring function of the class.
  virtual EvaluationResult EvaluateDataset(
      const VectorXi& k_list, const VectorXf& alpha_list,
      const Dataset& data, const SpMatrix& eval_by_user);

  virtual void Train(const Dataset& dataset) {}
  virtual void SetPrintTrainStats(const bool print_trainstats){};
  virtual void SetPrintResidualStats(
      const bool print_residualstats){};
  virtual void SetPrintVarStats(const bool print_varstats){};

  void init_matrix(MatrixXf* matrix, std::mt19937& gen,
                   const float adjusted_stdev) {
    std::normal_distribution<float> d(0, adjusted_stdev);
    for (int i = 0; i < matrix->size(); ++i) {
      *(matrix->data() + i) = d(gen);
    }
  };

  // Evaluate a single user.
  UserEvaluationResult EvaluateUser(const int num_items,
                                    const VectorXi& k_list,
                                    const VectorXf& all_scores,
                                    const SpVector& ground_truth,
                                    const SpVector& exclude);

  // Templated implementation for evaluating a dataset. Requires a
  // function that scores all items for a given user or history.
  template <typename F>
  EvaluationResult EvaluateDatasetInternal(
      const int num_items, const VectorXi& k_list,
      const VectorXf& alpha_list,
      const std::unordered_map<int, int>& user_to_ind,
      const Dataset& data, const SpMatrix& eval_by_user,
      F score_user_and_history) {
    std::mutex m;
    auto eval_by_user_iter = eval_by_user.begin();  // protected by m
    int num_ks = k_list.size();
    int num_users = eval_by_user.size();
    MatrixXf recall = MatrixXf::Zero(num_users, num_ks);
    MatrixXf ndcg = MatrixXf::Zero(num_users, num_ks);

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&] {
        while (true) {
          // Get a new user to work on.
          m.lock();
          if (eval_by_user_iter == eval_by_user.end()) {
            m.unlock();
            return;
          }
          int u = eval_by_user_iter->first;
          SpVector ground_truth = eval_by_user_iter->second;
          ++eval_by_user_iter;
          m.unlock();

          // Process the user.
          const SpVector& user_history = data.by_user().at(u);
          VectorXf scores = score_user_and_history(u, user_history);
          UserEvaluationResult this_metrics = this->EvaluateUser(
              num_items, k_list, scores, ground_truth, user_history);
          m.lock();
          // Update the metric.
          recall.row(user_to_ind.at(u)).noalias() +=
              this_metrics.recall;
          ndcg.row(user_to_ind.at(u)).noalias() += this_metrics.ndcg;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (auto& th : threads) {
      th.join();
    }

    EvaluationResult result = {k_list, alpha_list, recall, ndcg};
    return result;
  }
};

UserEvaluationResult Recommender::EvaluateUser(
    const int num_items, const VectorXi& k_list,
    const VectorXf& all_scores, const SpVector& ground_truth,
    const SpVector& exclude) {
  VectorXf scores = all_scores;
  for (uint64_t i = 0; i < exclude.size(); ++i) {
    assert(exclude[i].first < scores.size());
    scores[exclude[i].first] = std::numeric_limits<float>::lowest();
  }

  // Compute top-K ranking.
  int max_k = k_list.maxCoeff();
  std::vector<size_t> topk(scores.size());
  std::iota(topk.begin(), topk.end(), 0);
  std::nth_element(topk.begin(), topk.begin() + max_k, topk.end(),
                   [&scores](size_t i1, size_t i2) {
                     return scores[i1] > scores[i2];
                   });
  std::stable_sort(topk.begin(), topk.begin() + max_k,
                   [&scores](size_t i1, size_t i2) {
                     return scores[i1] > scores[i2];
                   });

  // Compute Recall@K.
  auto recall = [](int k, const std::set<int>& gt_set,
                   const std::vector<size_t>& topk) -> float {
    double result = 0.0;
    for (int i = 0; i < k; ++i) {
      if (gt_set.find(topk[i]) != gt_set.end()) {
        result += 1.0;
      }
    }
    return result / std::min<float>(k, gt_set.size());
  };

  // Compute nDCG@K.
  auto ndcg = [](int k, const std::set<int>& gt_set,
                 const std::vector<size_t>& topk) -> float {
    double result = 0.0;
    for (int i = 0; i < k; ++i) {
      if (gt_set.find(topk[i]) != gt_set.end()) {
        result += 1.0 / log2(i + 2.0);
      }
    }
    double norm = 0.0;
    for (int i = 0; i < std::min<int>(k, gt_set.size()); ++i) {
      norm += 1.0 / log2(i + 2.0);
    }
    return result / norm;
  };

  // Construct the set of positive items.
  std::set<int> gt_set;
  std::transform(
      ground_truth.begin(), ground_truth.end(),
      std::inserter(gt_set, gt_set.begin()),
      [](const std::pair<int, int>& p) { return p.first; });
  int num_ks = k_list.size();
  VectorXf recall_res(num_ks);
  VectorXf ndcg_res(num_ks);
  for (int i = 0; i < num_ks; ++i) {
    recall_res(i) = recall(k_list(i), gt_set, topk);
    ndcg_res(i) = ndcg(k_list(i), gt_set, topk);
  }

  UserEvaluationResult result = {recall_res, ndcg_res};
  return result;
}

EvaluationResult Recommender::EvaluateDataset(
    const VectorXi& k_list, const VectorXf& alpha_list,
    const Dataset& data, const SpMatrix& eval_by_user) {
  std::unordered_map<int, int> user_to_ind;
  return EvaluateDatasetInternal(
      data.max_item() + 1, k_list, alpha_list, user_to_ind, data,
      eval_by_user,
      [&](const int user_id, const SpVector& history) -> VectorXf {
        return Score(user_id, history);
      });
}

}  // namespace frecsys
