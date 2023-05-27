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

#pragma once

#include "Eigen/Core"
#include "Eigen/Dense"

namespace frecsys {

// Dense Vector/Matrix types
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>
    MatrixXf;

// Sparse Vector/Matrix types
using SpVector = std::vector<std::pair<int, int>>;
using SpMatrix = std::unordered_map<int, SpVector>;
}  // namespace frecsys
