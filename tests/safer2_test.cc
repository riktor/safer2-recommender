#include <chrono>
#include <unordered_map>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "frecsys/safer2.h"

using safer2_type = frecsys::SAFER2Recommender;
#define RECSYS_NAME "frecsys::SAFER2Recommender"

class SAFER2Test : public ::testing::Test {
protected:
  void SetUp() override {
    google::InstallFailureSignalHandler();

    flags_["embedding_dim"] = "8";
    flags_["unobserved_weight"] = "0.004";
    flags_["regularization"] = "0.004";
    flags_["regularization_exp"] = "1.0";
    flags_["stddev"] = "0.1";
    flags_["print_train_stats"] = "0";
    flags_["epochs"] = "10";
    flags_["alpha"] = "0.3";
    flags_["bandwidth"] = "0.15";
    flags_["use_snr"] = "0";
    flags_["use_epanechnikov"] = "0";

    // dataset path
    flags_["train_data"] = "tests/ml-1m/train.csv";
    flags_["test_train_data"] = "tests/ml-1m/validation_tr.csv";
    flags_["test_test_data"] = "tests/ml-1m/validation_te.csv";
  }
  std::unordered_map<std::string, std::string> flags_;
};

class SAFER2SNRTest : public ::testing::Test {
protected:
  void SetUp() override {
    google::InstallFailureSignalHandler();

    flags_["embedding_dim"] = "8";
    flags_["unobserved_weight"] = "0.004";
    flags_["regularization"] = "0.004";
    flags_["regularization_exp"] = "1.0";
    flags_["stddev"] = "0.1";
    flags_["print_train_stats"] = "0";
    flags_["epochs"] = "10";
    flags_["alpha"] = "0.3";
    flags_["bandwidth"] = "0.15";
    flags_["use_snr"] = "1";
    flags_["sampling_ratio"] = "0.5";
    flags_["use_epanechnikov"] = "0";

    // dataset path
    flags_["train_data"] = "tests/ml-1m/train.csv";
    flags_["test_train_data"] = "tests/ml-1m/validation_tr.csv";
    flags_["test_test_data"] = "tests/ml-1m/validation_te.csv";
  }
  std::unordered_map<std::string, std::string> flags_;
};

class SAFER2EPTest : public ::testing::Test {
protected:
  void SetUp() override {
    google::InstallFailureSignalHandler();

    flags_["embedding_dim"] = "8";
    flags_["unobserved_weight"] = "0.004";
    flags_["regularization"] = "0.004";
    flags_["regularization_exp"] = "1.0";
    flags_["stddev"] = "0.1";
    flags_["print_train_stats"] = "0";
    flags_["epochs"] = "10";
    flags_["alpha"] = "0.3";
    flags_["bandwidth"] = "0.7";
    flags_["use_snr"] = "0";
    flags_["use_epanechnikov"] = "1";

    // dataset path
    flags_["train_data"] = "tests/ml-1m/train.csv";
    flags_["test_train_data"] = "tests/ml-1m/validation_tr.csv";
    flags_["test_test_data"] = "tests/ml-1m/validation_te.csv";
  }
  std::unordered_map<std::string, std::string> flags_;
};

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
  EXPECT_LE(0.2, metrics.ndcg.colwise().mean()[2]);
}

TEST_F(SAFER2Test, TEST_SAFER2_ML1M) {
  google::InstallFailureSignalHandler();

  // Load the datasets
  frecsys::Dataset train(flags_.at("train_data"));
  frecsys::Dataset test_tr(flags_.at("test_train_data"));
  frecsys::Dataset test_te(flags_.at("test_test_data"));

  // Create the recommender.
  frecsys::SAFER2Recommender *recommender;
  recommender = new frecsys::SAFER2Recommender(
      std::atoi(flags_.at("embedding_dim").c_str()), train.max_user() + 1,
      train.max_item() + 1, std::atof(flags_.at("regularization").c_str()),
      std::atof(flags_.at("unobserved_weight").c_str()),
      std::atof(flags_.at("bandwidth").c_str()),
      std::atof(flags_.at("alpha").c_str()),
      std::atof(flags_.at("stddev").c_str()), 5, 1,
      std::atoi(flags_.at("use_epanechnikov").c_str()),
      std::atoi(flags_.at("use_snr").c_str()), 0.1, false, 1e-10, 1);
  ((frecsys::SAFER2Recommender *)recommender)
      ->SetPrintTrainStats(std::atoi(flags_.at("print_train_stats").c_str()));

  // Disable output buffer to see results without delay.
  setbuf(stdout, NULL);

  recommender->Initialize(train);
  float alpha = std::atof(flags_.at("alpha").c_str());

  // Train and evaluate.
  int num_epochs = std::atoi(flags_.at("epochs").c_str());
  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    auto time_train_start = std::chrono::steady_clock::now();
    recommender->Train(train);
    EXPECT_NEAR(alpha, recommender->GetMeanWeight(), 0.02);
    auto time_train_end = std::chrono::steady_clock::now();

    uint64_t train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_train_end - time_train_start)
                              .count();
    LOG(INFO) << fmt::format("Epoch: {0}, Timer: Train={1}", epoch, train_time);
  }
  evaluate(num_epochs, recommender, test_tr, test_te);

  delete recommender;
  return;
}

TEST_F(SAFER2SNRTest, TEST_SAFER2_ML1M) {
  google::InstallFailureSignalHandler();

  // Load the datasets
  frecsys::Dataset train(flags_.at("train_data"));
  frecsys::Dataset test_tr(flags_.at("test_train_data"));
  frecsys::Dataset test_te(flags_.at("test_test_data"));

  // Create the recommender.
  frecsys::SAFER2Recommender *recommender;
  recommender = new frecsys::SAFER2Recommender(
      std::atoi(flags_.at("embedding_dim").c_str()), train.max_user() + 1,
      train.max_item() + 1, std::atof(flags_.at("regularization").c_str()),
      std::atof(flags_.at("unobserved_weight").c_str()),
      std::atof(flags_.at("bandwidth").c_str()),
      std::atof(flags_.at("alpha").c_str()),
      std::atof(flags_.at("stddev").c_str()), 5, 1,
      std::atoi(flags_.at("use_epanechnikov").c_str()),
      std::atoi(flags_.at("use_snr").c_str()),
      std::atof(flags_.at("sampling_ratio").c_str()), false, 1e-10, 1);
  ((frecsys::SAFER2Recommender *)recommender)
      ->SetPrintTrainStats(std::atoi(flags_.at("print_train_stats").c_str()));

  // Disable output buffer to see results without delay.
  setbuf(stdout, NULL);

  recommender->Initialize(train);
  float alpha = std::atof(flags_.at("alpha").c_str());

  // Train and evaluate.
  int num_epochs = std::atoi(flags_.at("epochs").c_str());
  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    auto time_train_start = std::chrono::steady_clock::now();
    recommender->Train(train);
    EXPECT_NEAR(alpha, recommender->GetMeanWeight(), 0.02);
    auto time_train_end = std::chrono::steady_clock::now();

    uint64_t train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_train_end - time_train_start)
                              .count();
    LOG(INFO) << fmt::format("Epoch: {0}, Timer: Train={1}", epoch, train_time);
  }
  evaluate(num_epochs, recommender, test_tr, test_te);

  delete recommender;
  return;
}

TEST_F(SAFER2EPTest, TEST_SAFER2_ML1M) {
  google::InstallFailureSignalHandler();

  // Load the datasets
  frecsys::Dataset train(flags_.at("train_data"));
  frecsys::Dataset test_tr(flags_.at("test_train_data"));
  frecsys::Dataset test_te(flags_.at("test_test_data"));

  // Create the recommender.
  frecsys::SAFER2Recommender *recommender;
  recommender = new frecsys::SAFER2Recommender(
      std::atoi(flags_.at("embedding_dim").c_str()), train.max_user() + 1,
      train.max_item() + 1, std::atof(flags_.at("regularization").c_str()),
      std::atof(flags_.at("unobserved_weight").c_str()),
      std::atof(flags_.at("bandwidth").c_str()),
      std::atof(flags_.at("alpha").c_str()),
      std::atof(flags_.at("stddev").c_str()), 5, 1,
      std::atoi(flags_.at("use_epanechnikov").c_str()),
      std::atoi(flags_.at("use_snr").c_str()), 0.1, false, 1e-10, 1);
  ((frecsys::SAFER2Recommender *)recommender)
      ->SetPrintTrainStats(std::atoi(flags_.at("print_train_stats").c_str()));

  // Disable output buffer to see results without delay.
  setbuf(stdout, NULL);

  recommender->Initialize(train);
  float alpha = std::atof(flags_.at("alpha").c_str());

  // Train and evaluate.
  int num_epochs = std::atoi(flags_.at("epochs").c_str());
  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    auto time_train_start = std::chrono::steady_clock::now();
    recommender->Train(train);
    EXPECT_NEAR(alpha, recommender->GetMeanWeight(), 0.02);
    auto time_train_end = std::chrono::steady_clock::now();

    uint64_t train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_train_end - time_train_start)
                              .count();
    LOG(INFO) << fmt::format("Epoch: {0}, Timer: Train={1}", epoch, train_time);
  }
  evaluate(num_epochs, recommender, test_tr, test_te);

  delete recommender;
  return;
}
