load("//:bazel/frecsys.bzl", "frecsys_library")
frecsys_library("frecsys")

TEST_DEPS = [
    "@frecsys",
    "@com_gitlab_libeigen_eigen//:eigen",
    "@com_github_google_glog//:glog",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@fmt",
]

DEPS = [
    "@frecsys",
    "@com_gitlab_libeigen_eigen//:eigen",
    "@com_github_google_glog//:glog",
    "@fmt",
]

EXECUTABLE_DEPS = [
    "@frecsys",
    "@com_gitlab_libeigen_eigen//:eigen",
    "@com_github_google_glog//:glog",
    "@fmt",
]

TEST_COPTS = [
    "-g",
    "-Wall",
    "-O3",
    "-march=native",
    "-std=c++2a"
]

COPTS = [
    "-Wall",
    "-O3",
    "-march=native",
    "-std=c++2a",
]

TEST_DATA = [
    "tests/ml-1m/train.csv",
    "tests/ml-1m/validation_tr.csv",
    "tests/ml-1m/validation_te.csv"
]

LINKOPTS = ["-lpthread"]

[cc_test(
    name = test_filename.split("/")[-1].split('.')[0],
    timeout = "moderate",
    srcs = [test_filename],
    copts = TEST_COPTS,

    # This is the data set that is bundled for the testing.
    data = TEST_DATA,
    deps = TEST_DEPS,
    linkopts = LINKOPTS,
) for test_filename in glob([
    "tests/*_test.cc",
])]

cc_binary(
    name = "run_model",
    srcs = ["tools/run_model.cc", "tools/CLI11/CLI11.h"],
    copts = COPTS,
    deps = EXECUTABLE_DEPS,
    linkopts = LINKOPTS,
)


load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",

    # Specify the targets of interest.
    # For example, specify a dict of targets and any flags required to build.
    targets = [
      "frecsys", "run_model",
    ] + [test_filename.split("/")[-1].split('.')[0] for test_filename in glob(["tests/*_test.cc"])],
    #{
    #  "//:my_output_1": "--important_flag1 --important_flag2=true",
    #  "//:my_output_2": "",
    #},
    # No need to add flags already in .bazelrc. They're automatically picked up.
    # If you don't need flags, a list of targets is also okay, as is a single target string.
    # Wildcard patterns, like //... for everything, *are* allowed here, just like a build.
      # As are additional targets (+) and subtractions (-), like in bazel query https://docs.bazel.build/versions/main/query.html#expressions
    # And if you're working on a header-only library, specify a test or binary target that compiles it.
)
