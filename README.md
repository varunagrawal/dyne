# Dyne

Code for our Humanoids '22 paper "Proprioceptive State Estimation with Kinematic Chain Modeling".

## Repo Structure

- `src`: C++ source files including tests.
- `python`: Source files for building the python wrapper.
- `tests`: Python tests. Can evaluate `pytest tests`.
- `scripts`: Set of scripts to run various experiments.

## Development

### Dependencies

This project relies on

1. [gtsam](https://github.com/borglab/gtsam)
2. [gtdynamics](https://github.com/borglab/gtdynamics)

Please refer to their README for instructions on how to compile and build the libraries.

### Compilation

We use `CMake` to configure the build directives, and `make` to build the library.

```sh
mkdir build && cd build
cmake ..
make -j8        # build
make -j8 check  # run tests
```

### Python

Running CMake with the `DYNE_BUILD_PYTHON_WRAPPER=ON` flag will build the python wrapper.

This will create a new directory `dyne` in the root folder with the built python module.
You can now run `pip install -e .` to install the python package.

## State Estimators

We provide the following estimators:

1. `BaseImu`: This is an IMU-only dead-reckoning based estimator which relies on IMU preintegration. Liable to drift due to varying bias.
2. `Bloesch`: This is a factor-graph based implementation of the state estimator proposed by Bloesch et. al. in "Consistent Fusion of Leg Kinematics and IMU".
3. `Dyne`: Our state estimator proposed in the Humanoids 2022 paper.
