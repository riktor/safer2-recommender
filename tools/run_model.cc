// Copyright 2022 ************
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "CLI11/CLI11.h"
#include <chrono>
#include <fmt/core.h>
#include <glog/logging.h>
#include <string>

#include "Eigen/Core"
#include "Eigen/Dense"

#include "frecsys/cvar_mf.h"
#include "frecsys/erm_mf.h"
#include "frecsys/ials.h"
#include "frecsys/ialspp.h"
#include "frecsys/safer2.h"
#include "frecsys/safer2pp.h"

template <typename F>
void evaluate(int epoch, F recommender, frecsys::Dataset &exclude,
              frecsys::Dataset &test) {
  Eigen::VectorXi k_list = Eigen::VectorXi::Zero(5);
  Eigen::VectorXf alpha_list = Eigen::VectorXf::Zero(9);
  k_list << 5, 10, 20, 50, 100;
  alpha_list << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
  frecsys::EvaluationResult metrics =
      recommender->EvaluateDataset(k_list, alpha_list, exclude, test.by_user());
  LOG(INFO) << "Epoch " << epoch << ":";
  metrics.show();
}

frecsys::Recommender *get_model(const std::string model_name,
                                const int num_users, const int num_items,
                                CLI::App &app) {
  frecsys::Recommender *recommender = nullptr;
  if (model_name == "ials") {
    recommender = new frecsys::IALSRecommender(
        app.get_option("--dim")->as<int>(), num_users, num_items,
        app.get_option("--l2_reg")->as<float>(),
        app.get_option("--l2_reg_exp")->as<float>(),
        app.get_option("--uobs_weight")->as<float>(),
        app.get_option("--stdev")->as<float>(),
        app.get_option("--alpha")->as<float>(),
        app.get_option("--use_cg")->as<bool>(),
        app.get_option("--cg_error_tolerance")->as<float>(),
        app.get_option("--cg_max_iterations")->as<int>());
  } else if (model_name == "ialspp") {
    recommender = new frecsys::IALSppRecommender(
        app.get_option("--dim")->as<int>(), num_users, num_items,
        app.get_option("--l2_reg")->as<float>(),
        app.get_option("--l2_reg_exp")->as<float>(),
        app.get_option("--uobs_weight")->as<float>(),
        app.get_option("--stdev")->as<float>(),
        app.get_option("--alpha")->as<float>(),
        app.get_option("--block_size")->as<int>());
  } else if (model_name == "safer2") {
    recommender = new frecsys::SAFER2Recommender(
        app.get_option("--dim")->as<int>(), num_users, num_items,
        app.get_option("--l2_reg")->as<float>(),
        app.get_option("--uobs_weight")->as<float>(),
        app.get_option("--bandwidth")->as<float>(),
        app.get_option("--alpha")->as<float>(),
        app.get_option("--stdev")->as<float>(),
        app.get_option("--xi_iterations")->as<int>(),
        app.get_option("--pd_iterations")->as<int>(),
        app.get_option("--use_epanechnikov")->as<bool>(),
        app.get_option("--use_snr")->as<bool>(),
        app.get_option("--sampling_ratio")->as<float>(),
        app.get_option("--use_cg")->as<bool>(),
        app.get_option("--cg_error_tolerance")->as<float>(),
        app.get_option("--cg_max_iterations")->as<int>());
  } else if (model_name == "safer2pp") {
    recommender = new frecsys::SAFER2ppRecommender(
        app.get_option("--dim")->as<int>(), num_users, num_items,
        app.get_option("--l2_reg")->as<float>(),
        app.get_option("--uobs_weight")->as<float>(),
        app.get_option("--bandwidth")->as<float>(),
        app.get_option("--alpha")->as<float>(),
        app.get_option("--stdev")->as<float>(),
        app.get_option("--xi_iterations")->as<int>(),
        app.get_option("--pd_iterations")->as<int>(),
        app.get_option("--use_epanechnikov")->as<bool>(),
        app.get_option("--use_snr")->as<bool>(),
        app.get_option("--sampling_ratio")->as<float>(),
        app.get_option("--block_size")->as<int>());
  } else if (model_name == "erm_mf") {
    recommender = new frecsys::ERMMFRecommender(
        app.get_option("--dim")->as<int>(), num_users, num_items,
        app.get_option("--l2_reg")->as<float>(),
        app.get_option("--uobs_weight")->as<float>(),
        app.get_option("--stdev")->as<float>(),
        app.get_option("--alpha")->as<float>(),
        app.get_option("--use_cg")->as<bool>(),
        app.get_option("--cg_error_tolerance")->as<float>(),
        app.get_option("--cg_max_iterations")->as<int>());
  } else if (model_name == "cvar_mf") {
    recommender = new frecsys::CVaRMFRecommender(
        app.get_option("--dim")->as<int>(), num_users, num_items,
        app.get_option("--l2_reg")->as<float>(),
        app.get_option("--uobs_weight")->as<float>(),
        app.get_option("--alpha")->as<float>(),
        app.get_option("--stepsize")->as<float>(),
        app.get_option("--stdev")->as<float>());
  }
  recommender->SetPrintResidualStats(
      app.get_option("--print_residual_stats")->as<bool>());
  recommender->SetPrintVarStats(
      app.get_option("--print_var_stats")->as<bool>());
  recommender->SetPrintTrainStats(
      app.get_option("--print_train_stats")->as<bool>());
  return recommender;
}

int main(int argc, char *argv[]) {
  CLI::App app{"frecsys experimentation utility"};

  // Options
  bool print_evaluation_stats = false;
  app.add_option("--print_evaluation_stats", print_evaluation_stats,
                 "Verbosity of evaluation result per epoch");

  app.add_option("-d,--dim", "Embedding dimensionality of MF models")
      ->default_val(8);

  app.add_option("--uobs_weight",
                 "Weight of norm regularisation for recovered matrix")
      ->default_val(0.1);

  app.add_option("-r,--l2_reg", "Base weight of L2 regularisation")
      ->default_val(0.002);

  app.add_option("--l2_reg_exp",
                 "Exponent of Frequency-based L2 regularisation")
      ->default_val(1.0);

  app.add_option(
         "-s,--stdev",
         "Standard deviation of normal noises for parameter initialisation")
      ->default_val(0.1);

  app.add_option("--print_train_stats", "Verbosity of training statistics")
      ->default_val(true);

  app.add_option("--print_test_results", "Verbosity of testing results")
      ->default_val(false);

  app.add_option("--print_residual_stats",
                 "Verbosity of residual statistics for SAFER")
      ->default_val(false);

  app.add_option("--print_var_stats", "Verbosity of VaR statistics for SAFER")
      ->default_val(false);

  app.add_option("--cg_error_tolerance",
                 "Error tolerance for conjugate gradient")
      ->default_val(1e-10);
  app.add_option("--cg_max_iterations",
                 "Max number of iterations for conjugate gradient")
      ->default_val(100);

  app.add_option("--use_cg", "Use conjugate gradients")->default_val(false);

  app.add_option("--block_size", "Block size for sub-space optimisation")
      ->default_val(64);

  // Options for SAFER2
  app.add_option("--alpha", "Quantile level for (C)VaR.")->default_val(0.3);
  app.add_option("--bandwidth", "Bandwidth of kernel functions")
      ->default_val(1.0);
  app.add_option("--stepsize", "Stepsize of CVaR-MF")->default_val(0.1);

  app.add_option("--xi_iterations",
                 "Number of iterations for CtS-VaR estimation")
      ->default_val(5);

  app.add_option("--sampling_ratio",
                 "Sampling ratio of Stochastic Newton-Raphson method for "
                 "CtS-VaR estimation")
      ->default_val(0.1);

  app.add_option("--pd_iterations",
                 "Number of iterations for primal-dual algorithm")
      ->default_val(1);

  app.add_option("--use_epanechnikov",
                 "Use Epanechnikov kernel (default Gaussian kernel)")
      ->default_val(false);

  app.add_option("--use_snr", "Use sub-sampled Newton-Raphson")
      ->default_val(false);

  int epochs = 50;
  app.add_option("-e,--epoch", epochs, "Number of epochs");

  std::map<std::string, std::string> model_map{
      {"ials", "ials"},         {"ialspp", "ialspp"},   {"safer2", "safer2"},
      {"safer2pp", "safer2pp"}, {"cvar_mf", "cvar_mf"}, {"erm_mf", "erm_mf"}};
  std::string model_name;
  app.add_option("-n,--model_name", model_name,
                 "Model name in [ials, cvar_mf, safer2, erm_mf]")
      ->required()
      ->check(CLI::CheckedTransformer(model_map, CLI::ignore_case));

  std::string train_data;
  app.add_option("--train_data", train_data, "Path of the training data file")
      ->required()
      ->check(CLI::ExistingFile);

  std::string test_train_data;
  app.add_option("--test_train_data", test_train_data,
                 "Path of the training data file in prediction phase")
      ->required()
      ->check(CLI::ExistingFile);

  std::string test_test_data;
  app.add_option("--test_test_data", test_test_data,
                 "Path of the testing data file in prediction phase")
      ->required()
      ->check(CLI::ExistingFile);

  CLI11_PARSE(app, argc, argv);

  // Load the datasets
  frecsys::Dataset train(train_data);
  frecsys::Dataset test_tr(test_train_data);
  frecsys::Dataset test_te(test_test_data);

  frecsys::Recommender *recommender =
      get_model(model_name, train.max_user() + 1, train.max_item() + 1, app);

  // Disable output buffer to see results without delay.
  setbuf(stdout, NULL);
  // Train and evaluate.

  if (model_name == "cvar_mf") {
    ((frecsys::CVaRMFRecommender *)recommender)->Initialize(train);
  }
  if (model_name == "safer2") {
    ((frecsys::SAFER2Recommender *)recommender)->Initialize(train);
  }
  if (model_name == "safer2pp") {
    ((frecsys::SAFER2ppRecommender *)recommender)->Initialize(train);
  }
  if (model_name == "erm_mf") {
    ((frecsys::ERMMFRecommender *)recommender)->Initialize(train);
  }
  for (int epoch = 0; epoch < epochs; ++epoch) {
    auto time_train_start = std::chrono::steady_clock::now();
    recommender->Train(train);
    auto time_train_end = std::chrono::steady_clock::now();

    uint64_t train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_train_end - time_train_start)
                              .count();
    LOG(INFO) << fmt::format("Epoch: {0}, Timer: Train={1}", epoch, train_time);
    if (print_evaluation_stats) {
      evaluate(epoch, recommender, test_tr, test_te);
    }
  }
  LOG(INFO) << "Validation Results";
  evaluate(epochs, recommender, test_tr, test_te);
  return 0;
}
