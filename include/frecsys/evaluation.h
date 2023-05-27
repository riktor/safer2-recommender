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

#include <fmt/core.h>
#include <glog/logging.h>
#include <Eigen/Dense>
#include <algorithm>
#include <vector>
#include "frecsys/types.h"

namespace frecsys {

// Evaluation results for a single user.
struct UserEvaluationResult {
  const VectorXf recall;
  const VectorXf ndcg;
};

// Evaluation results for testing users.
struct EvaluationResult {
  const VectorXi k_list;
  const VectorXf alpha_list;
  const MatrixXf recall;
  const MatrixXf ndcg;

  // Generate a formatted string for measure@K.
  std::string format(std::string measure_name,
                     VectorXf measurements) const {
    assert(k_list.size() == measurements.size());

    std::stringstream ss;

    for (int i = 0; i < k_list.size(); i++) {
      int k = k_list[i];
      ss << fmt::format("{0}@{1}={2:.4f}", measure_name, k,
                        measurements[i]);
      if (i != k_list.size() - 1) {
        ss << " ";
      }
    }
    return ss.str();
  }

  // Emit logs of ranking measures.
  void show() const {
    LOG(INFO) << format("Mean Rec", this->recall.colwise().mean());
    LOG(INFO) << format("Mean NDCG", this->ndcg.colwise().mean());

    MatrixXf ndcg_cvar(k_list.size(), alpha_list.size());
    MatrixXf rec_cvar(k_list.size(), alpha_list.size());
    for (int i = 0; i < k_list.size(); i++) {
      ndcg_cvar.row(i).noalias() =
          cvar(this->ndcg.transpose().row(i));
      rec_cvar.row(i).noalias() =
          cvar(this->recall.transpose().row(i));
    }
    for (int i = 0; i < alpha_list.size(); i++) {
      LOG(INFO) << format(
          fmt::format("Rec CVaR (q={0:.2f})", alpha_list[i]),
          rec_cvar.col(i));
      LOG(INFO) << format(
          fmt::format("NDCG CVaR (q={0:.2f})", alpha_list[i]),
          ndcg_cvar.col(i));
    }
  }

  VectorXf cvar(VectorXf measurements) const {
    std::vector<float> ms(measurements.data(),
                          measurements.data() + measurements.rows());

    std::sort(ms.begin(), ms.end());
    int counter = 0;
    VectorXf cvars = VectorXf::Zero(alpha_list.size());
    float accs = 0;
    for (int i = 0; i < ms.size(); i++) {
      accs += ms.at(i);
      for (int j = counter; j < alpha_list.size(); j++) {
        int pos = ms.size() * alpha_list[j];
        if (pos == i) {
          cvars[counter] = accs / (i + 1);
          counter++;
        }
      }
    }
    return cvars;
  }
};

}  // namespace frecsys
