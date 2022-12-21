"""
Various utilities.
"""

import re
from pathlib import Path
from typing import Dict, Iterable, Union

import gtsam
import numpy as np
import yaml
from evo.tools import file_interface
from loguru import logger
from tabulate import tabulate

from lrse import datasets, metrics
from lrse.state_estimator import RobotImu


def get_limits(lower, upper, padding=0.1):
    """
    Add padding to the lower and upper limits.

    Args:
        lower: Lower limit
        upper: Upper limit
        padding: The amount of padding to add as a fractional percentage [0, 1]
    """
    upper_padding = padding * upper

    if upper > 0:
        upper += upper_padding
    elif upper == 0:
        upper += padding
    else:
        upper -= upper_padding

    lower_padding = padding * lower
    if lower > 0:
        lower -= lower_padding
    elif lower == 0:
        lower -= padding
    else:
        lower += lower_padding

    return (lower, upper)


def estimate_initial_biases(measured_acc, measured_omega):
    """Estimate initial bias means and variances from when the IMU is stationary."""
    omega_mean = np.mean(measured_omega, axis=0)
    # omega_var = np.var(measured_omega, axis=0)
    acc_mean = np.mean(measured_acc, axis=0)
    # acc_var = np.var(measured_acc, axis=0)
    return acc_mean, omega_mean


def yaml_loader():
    """
    Custom yaml loader definition which is more generic.
    """
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(
            u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X), list(u'-+0123456789.'))
    return loader


def load_imu_params(imu_yaml: Union[str, Path]) -> Dict:
    """Helper method to load the IMU parameters from the YAML file."""
    with open(imu_yaml) as imu_data:
        imu_params = yaml.load(imu_data, Loader=yaml_loader())

    # update the IMU params with nicer values
    body_P_sensor = gtsam.Pose3(gtsam.Rot3.RzRyRx(*imu_params["pose"][3:]),
                                gtsam.Point3(*imu_params["pose"][0:3]))
    imu_params["pose"] = body_P_sensor

    imu_params["freq"] = imu_params["rate_hz"]

    return imu_params


def robot_imu_from_yaml(imu_yaml_file: Path):
    """
    Method to get RobotImu from
    a YAML file containing the configuration data.

    Args:
        imu_yaml_file: YAML file with all the IMU configuration data.
    """
    imu_params = load_imu_params(imu_yaml_file)
    return RobotImu(
        imu_params["rate_hz"],  #
        imu_params["gyroscope"]["noise"]["stddev"],  #
        imu_params["accelerometer"]["noise"]["stddev"],  #
        imu_params["pose"])


def add_stats(ape, rpe, stats=None, units=""):
    """Add the `ape` statistics and the `rpe` statistics to a single dict for tabulate."""
    if stats is None:
        stats = {}

    for metric, value in ape.items():
        if metric not in stats:
            stats[metric] = [units, value]
        else:
            stats[metric].append(value)

    for metric, value in rpe.items():
        if metric not in stats:
            stats[metric] = [value]
        else:
            stats[metric].append(value)

    return stats


def get_table(stats: Dict, headers: Iterable, tablefmt="github"):
    """
    Use `tabulate` to get a pretty table of the statistics.
    Args:
        stats: Dict of statistics to get as a table.
        headers: The table headers.
        tablefmt: The format to use, e.g. `github`, `latex`.
    """
    stats_table = [[k, *v] for k, v in stats.items()]
    return tabulate(stats_table,
                    headers=headers,
                    tablefmt=tablefmt,
                    floatfmt=".5f")


def save_table(stats: Dict[str, Dict],
               headers: Iterable = ("Metric", "APE Stats", "RPE Stats"),
               tablefmt="github",
               output_file="table.md"):
    """Generate statistics table and save to `output_file`."""
    with open(output_file, "w") as of:
        table = get_table(stats, headers, tablefmt)
        of.write(table + "\n\n")

    logger.info("Wrote table data to {}".format(output_file))


def print_table(stats: Dict,
                headers: Iterable = ("Metric", "Units", "APE", "RPE"),
                tablefmt="github"):
    """
    Print dict in a pretty table format.
    """
    table = get_table(stats, headers, tablefmt)
    print(table)


def write_evo(traj, filename="output.traj"):
    """
    Write values to output file in TUM format that can be parsed by `evo`.
    """
    file_interface.write_tum_trajectory_file(filename, traj)
    logger.info("Wrote trajectory to {}".format(filename))


def write_ground_truth(dataset, T, dt, filename, sampling_rate=1):
    """
    Convenience funciton to write the ground truth data in TUM format
    to generate the evo reference trajectory.
    """
    values = dataset.ground_truth_as_values(sampling_rate)
    traj = metrics.get_evo_trajectory(values, T, dt, sampling_rate)
    write_evo(traj, filename=filename)
    return traj


def save_results(estimator,
                 dataset: datasets.Dataset,
                 results: gtsam.Values,
                 output_dir: str,
                 estimated_name: str,
                 ground_truth_name: str = "ground_truth",
                 sampling_rate: int = 1,
                 estimator_rate=None,
                 align_trajectories: bool = False):
    """Compute and save estimator results on dataset to `output_dir`."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if estimator_rate is None:
        estimator_rate = estimator.rate()

    values: gtsam.Values = dataset.ground_truth_as_values(sampling_rate)
    reference_traj = metrics.get_evo_trajectory(values, dataset.timestamps,
                                                sampling_rate)

    # Get the trajectory in evo format and write to file
    estimated_traj = metrics.get_evo_trajectory(results, dataset.timestamps,
                                                estimator_rate)

    if align_trajectories:
        reference_traj, estimated_traj, _ = metrics.align_trajectories(
            reference_traj, estimated_traj, estimator_rate)

    # Write out trajectories to file
    write_evo(estimated_traj,
              filename=output_dir / "{0}.traj".format(estimated_name))
    write_evo(reference_traj,
              filename=output_dir / f"{ground_truth_name}.traj")

    ape_stats, rpe_stats = metrics.evaluate(reference_traj, estimated_traj, delta=estimator_rate)

    return ape_stats, rpe_stats
