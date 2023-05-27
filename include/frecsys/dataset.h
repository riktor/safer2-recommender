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

#include <fstream>

#include <glog/logging.h>
#include "frecsys/types.h"

namespace frecsys {

class Dataset {
public:
  explicit Dataset(const std::string& filename);
  const SpMatrix& by_user() const {
    return by_user_;
  }
  const SpMatrix& by_item() const {
    return by_item_;
  }
  const int max_user() const {
    return max_user_;
  }
  const int max_item() const {
    return max_item_;
  }
  const int num_tuples() const {
    return num_tuples_;
  }

  void merge(const Dataset& dataset) {
    for (auto& entry : dataset.by_user()) {
      auto v = entry.second;
      // LOG(INFO) << "user id: " << entry.first;
      auto target = by_user_.at(entry.first);
      for (int i = 0; i < v.size(); i++) {
        target.push_back(v[i]);
      }
    }

    for (auto& entry : dataset.by_item()) {
      auto v = entry.second;
      // LOG(INFO) << "item id: " << entry.first;
      auto target = by_item_.at(entry.first);
      for (int i = 0; i < v.size(); i++) {
        target.push_back(v[i]);
      }
    }
  }

private:
  SpMatrix by_user_;
  SpMatrix by_item_;
  int max_user_;
  int max_item_;
  int num_tuples_;
};

Dataset::Dataset(const std::string& filename) {
  max_user_ = -1;
  max_item_ = -1;
  num_tuples_ = 0;

  std::ifstream infile(filename);
  std::string line;

  // Discard header.
  assert(std::getline(infile, line));

  // Read the data.
  while (std::getline(infile, line)) {
    int pos = line.find(',');
    int user = std::atoi(line.substr(0, pos).c_str());
    int item = std::atoi(line.substr(pos + 1).c_str());
    by_user_[user].push_back({item, num_tuples_});
    by_item_[item].push_back({user, num_tuples_});
    max_user_ = std::max(max_user_, user);
    max_item_ = std::max(max_item_, item);
    ++num_tuples_;
  }

  LOG(INFO) << "max_user=" << max_user()
            << "\tmax_item=" << max_item()
            << "\tdistinct user=" << by_user_.size()
            << "\tdistinct item=" << by_item_.size()
            << "\tnum_tuples=" << num_tuples();
};

}  // namespace frecsys
