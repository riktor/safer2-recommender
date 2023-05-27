# SAFER<sub>2</sub>
This is the implementation of Smoothing Approach for Efficient Risk-averse Recommender (**SAFER<sub>2</sub>**), a safe collaborative filtering method.

```tex
@article{togashi2023safe,
 title={Safe Collaborative Filtering},
 author={Togashi, Riku and Oka, Tatsushi and Ohsaka, Naoto and Morimura, Tetsuro},
 journal={arXiv preprint arXiv:},
 year={2023}
}
```

## Setup

### Build all-in-one docker image.
Build docker image and generate datasets inside the image.
```sh
$ sh build_image.sh
```

### Build executable without docker.
The code can be used without docker.
See Dockerfile for an example installation, which contains the minimum setup instructions for a debian-based image (slim).

Build executable locally. Need to install [bazel](https://github.com/bazelbuild/bazel) and gcc/g++ (versions from 9 through 12).
```sh
$ bazel build run_model
```

Create convenient symlinks (optional).
```sh
$ ln -s $(bazel info bazel-bin) bazel-bin
```

Install dependency.
```sh
$ pip install -r scripts/requirements.txt
```

Download and generate datasets.
```sh
$ python scripts/generate_data.py
```

### Testing
Run tests manually for iALS, ERM-MF, CVaR-MF and SAFER<sub>2</sub>.
```sh
$ bazel test //... --test_output=all
```
These tests run automatically through github workflow.
See `.github/workflow/bazel.yaml` for details.


## Run Models inside Docker Container
### ML-1M
SAFER<sub>2</sub>
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name safer2 --dim 32 --uobs_weight 0.004 --alpha 0.3 --l2_reg 0.004 --use_snr 0 --xi_iterations 5 --pd_iterations 1 --stdev 0.1 --print_train_stats 1 --epoch 50 --train_data ml-1m/train.csv --test_train_data ml-1m/validation_tr.csv --test_test_data ml-1m/validation_te.csv --print_evaluation_stats 1 --bandwidth 0.15
```

iALS
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name ials --dim 32 --uobs_weight 0.2 --l2_reg 0.006 --stdev 0.1 --print_train_stats 1 --epoch 50 --train_data ml-1m/train.csv --test_train_data ml-1m/validation_tr.csv --test_test_data ml-1m/validation_te.csv --print_evaluation_stats 1
```

ERM-MF
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name erm_mf --dim 32 --uobs_weight 0.004 --alpha 0.3 --l2_reg 0.005 --stdev 0.1 --print_train_stats 1 --epoch 50 --train_data ml-1m/train.csv --test_train_data ml-1m/validation_tr.csv --test_test_data ml-1m/validation_te.csv --print_evaluation_stats 1
```

CVaR-MF
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name cvar_mf --dim 32 --uobs_weight 0.008 --alpha 0.3 --l2_reg 0.002 --stepsize 0.4 --stdev 0.1 --print_train_stats 1 --epoch 300 --train_data ml-1m/train.csv --test_train_data ml-1m/validation_tr.csv --test_test_data ml-1m/validation_te.csv --print_evaluation_stats 1
```

### ML-20M
SAFER<sub>2</sub>
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name safer2 --dim 256 --uobs_weight 0.002 --alpha 0.3 --l2_reg 0.002 --stdev 0.1 --print_train_stats 1 --epoch 50 --train_data ml-20m/train.csv --test_train_data ml-20m/validation_tr.csv --test_test_data ml-20m/validation_te.csv --bandwidth 0.18 --pd_iterations 1 --xi_iterations 5 --print_evaluation_stats 1 --use_snr 1 --sampling_ratio 0.1
```

iALS
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name safer2 --dim 256 --uobs_weight 0.1 --l2_reg 0.003 --stdev 0.1 --print_train_stats 1 --epoch 50 --train_data ml-20m/train.csv --test_train_data ml-20m/validation_tr.csv --test_test_data ml-20m/validation_te.csv --print_evaluation_stats 1
```

ERM-MF
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name erm_mf --dim 256 --uobs_weight 0.002 --alpha 0.3 --l2_reg 0.003 --stdev 0.1 --print_train_stats 1 --epoch 50 --train_data ml-20m/train.csv --test_train_data ml-20m/validation_tr.csv --test_test_data ml-20m/validation_te.csv --print_evaluation_stats 1 --use_snr 1 --sampling_ratio 0.1
```

CVaR-MF
```
docker run -it frecsys_box:latest bazel-bin/run_model --model_name cvar_mf --dim 256 --uobs_weight 0.0009 --alpha 0.3 --l2_reg 0.0004 --stepsize 0.4 --stdev 0.1 --print_train_stats 1 --epoch 1000 --train_data ml-20m/train.csv --test_train_data ml-20m/validation_tr.csv --test_test_data ml-20m/validation_te.csv --print_evaluation_stats 1
```

### MSD
SAFER<sub>2</sub>
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name safer2 --dim 512 --uobs_weight 0.0004 --alpha 0.3 --l2_reg 0.0012 --l2_reg_exp 1.0 --stdev 0.1 --print_train_stats 1 --epoch 50 --train_data msd/train.csv --test_train_data msd/validation_tr.csv --test_test_data msd/validation_te.csv --bandwidth 0.1 --pd_iterations 1 --xi_iterations 5 --print_evaluation_stats 1 --use_snr 1 --sampling_ratio 0.1
```

iALS
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name ials --dim 512 --uobs_weight 0.05 --l2_reg 0.002 --stdev 0.1 --print_train_stats 1 --epoch 50 --train_data msd/train.csv --test_train_data msd/validation_tr.csv --test_test_data msd/validation_te.csv --print_evaluation_stats 1
```

ERM-MF
```sh
docker run -it frecsys_box:latest bazel-bin/run_model --model_name erm-mf --dim 512 --uobs_weight 0.0004 --alpha 0.3 --l2_reg 0.0012 --l2_reg_exp 1.0 --stdev 0.1 --print_train_stats 1 --epoch 50 --train_data msd/train.csv --test_train_data msd/validation_tr.csv --test_test_data msd/validation_te.csv --print_evaluation_stats 1
```

CVaR-MF
```
docker run -it frecsys_box:latest bazel-bin/run_model --model_name cvar_mf --dim 512 --uobs_weight 0.004 --alpha 0.3 --l2_reg 0.0004 --stepsize 0.4 --stdev 0.1 --print_train_stats 1 --epoch 1000 --train_data msd/train.csv --test_train_data msd/validation_tr.csv --test_test_data msd/validation_te.csv --print_evaluation_stats 1
```


## Directory Structure
Following is the directory structure of this repository,
which may be helpful to read through the code.

```
.
├── build_image.sh          (for building reproducible docker image)
├── Dockerfile
├── .bazelrc                (bazel configuration)
│── WORKSPACE               (for dependency)
├── BUILD                   (for C++ compile options and linking)
├── bazel
│   └── frecsys.bzl
├── 3rdparty
│   └── eigen.BUILD         (for using the latest Eigen)
├── include                 (**header-only implementation**)
│   └── frecsys
│       ├── types.h         (type definition)
│       ├── dataset.h       (struct definition of dataset)
│       ├── evaluation.h    (struct definition of evaluation results)
│       ├── recommender.h   (definition of the abstract base class)
│       ├── ials.h          (implementation of iALS)
│       ├── safer2.h        (implementation of SAFER2)
│       ├── erm_mf.h        (implementation of ERM-MF)
│       ├── cvar_mf.h       (implementation of CVaR-MF)
│       ├── ialspp.h        (implementation of iALS++)
│       └── safer2pp.h      (implementation of SAFER2++)
├── README.md
├── scripts                 (**miscellaneous scripts**)
│   ├── requirements.txt    (python dependency)
│   └── generate_data.py    (script for preparing datasets)
├── tests
│   ├── ials_test.cc        (integration test of iALS)
│   ├── safer2_test.cc      (integration test of SAFER2)
│   ├── erm_mf_test.cc      (integration test of ERM-MF)
│   ├── cvar_mf_test.cc     (integration test of CVaR-MF)
│   ├── ialspp_test.cc      (integration test of iALS++)
│   ├── safer2pp_test.cc    (integration test of SAFER2++)
│   ├── ml-1m               (sample dataset for integration testing)
│   │   ├── train.csv
│   │   ├── validation_te.csv
│   └── └── validation_tr.csv
├── tools                   (**executable implementation**)
│   ├── CLI11               (dependency for CLI)
│   │   └── CLI11.h
└── └── run_model.cc        (implementation of CLI)
```
