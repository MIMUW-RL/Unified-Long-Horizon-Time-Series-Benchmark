import h5py
import numpy as np
import torch

from problem_specs import ProblemSpec


def get_obs_dim(problem_spec_name):
    if problem_spec_name == "cheetah_medium_dense_1000":
        return 17
    elif problem_spec_name == "cheetah_medium_dense_500":
        return 17
    if (problem_spec_name == "large_cheetah_medium_dense_1000") or (problem_spec_name == "large_cheetah_medium_deter_1000"):
        return 17
    elif problem_spec_name == "large_cheetah_medium_dense_1000_shorter":
        return 17
    elif problem_spec_name == "large_cheetah_medium_dense_500":
        return 17
    elif problem_spec_name == "hopper_medium_dense_1000":
        return 11
    elif problem_spec_name == "hopper_medium_dense_500":
        return 11
    elif (problem_spec_name == "large_hopper_medium_dense_1000") or (problem_spec_name == "large_hopper_medium_deter_1000"):
        return 11
    elif problem_spec_name == "large_hopper_medium_dense_500":
        return 11
    elif problem_spec_name == "ks100_500":
        return 100
    elif problem_spec_name == "ks100_1000":
        return 100
    elif problem_spec_name == "large_ks100_500":
        return 100
    elif problem_spec_name == "large_ks100_1000":
        return 100
    elif problem_spec_name == "large_turbulence_1000":
        return 256
    elif problem_spec_name == "walker_medium_dense_1000":
        return 17
    elif problem_spec_name == "walker_medium_dense_500":
        return 17
    elif (problem_spec_name == "large_walker_medium_dense_1000") or (problem_spec_name == "large_walker_medium_deter_1000"):
        return 17
    elif problem_spec_name == "large_walker_medium_dense_500":
        return 17
    elif problem_spec_name == "electricity":
        return 1
    elif problem_spec_name == "electricity_short":
        return 1
    elif problem_spec_name == "m4_247":
        return 1
    elif problem_spec_name == "m4_short":
        return 1
    elif problem_spec_name == "tourism_228":
        return 1
    elif problem_spec_name == "tourism_short":
        return 1
    elif problem_spec_name == "large_mackey_glass_2000":
        return 1
    elif problem_spec_name == "large_mackey_glass_2000_shorter":
        return 1
    elif problem_spec_name == "large_lorenz_2000":
        return 3
    elif problem_spec_name == "large_mso_2000":
        return 1
    elif problem_spec_name == "large_sspiral_2000":
        return 2
    elif problem_spec_name == "ett_small":
        return 1
    elif problem_spec_name == "ett_small_short":
        return 1
    elif problem_spec_name == "ett_long":
        return 1
    elif problem_spec_name == "ett_long_short":
        return 1
    elif problem_spec_name == "large_cahn_hillard_1000":
        return 256
    elif problem_spec_name == "large_lotka_2000":
        return 2
    elif problem_spec_name == "pems_bay":
        return 325
    elif problem_spec_name == "pems_bay_short":
        return 325
    elif problem_spec_name == "weather":
        return 21
    else:
        raise ValueError(
            f"obs dim unknown for problem spec: {problem_spec_name}"
        )


def problem_spec_from_hdf5_standard(
    file_path,
    trunc=None,
    shorter=None,
):
    f = h5py.File(file_path, "r")

    train_dataset = np.array(f["train_data"])
    test_dataset = np.array(f["test_data"])
    train_timestamps = np.array(f["train_timestamps"])[
        :, : train_dataset.shape[1]
    ]
    test_timestamps = np.array(f["test_timestamps"])[
        :, : test_dataset.shape[1]
    ]
    f.close()

    problem_spec = ProblemSpec(
        train_dataset=torch.from_numpy(train_dataset).float(),
        test_dataset=torch.from_numpy(test_dataset).float(),
        train_timestamps=torch.from_numpy(train_timestamps).float(),
        test_timestamps=torch.from_numpy(test_timestamps).float(),
        train_out_frac=None,
        trunc=trunc,
        less_training_trajectories=shorter,
    )

    return problem_spec

def weather():
    return problem_spec_from_hdf5_standard(
        file_path="./data/weather/dataset.hdf5",
        trunc=1000,
    )

def ett_small():
    return problem_spec_from_hdf5_standard(
        file_path="./data/ett/dataset.hdf5",
        trunc=1440,
    )

def ett_small_short():
    return problem_spec_from_hdf5_standard(
        file_path="./data/ett/dataset.hdf5",
        trunc=1440,
        shorter=1000,
    )

def m4_short():
    return problem_spec_from_hdf5_standard(
        file_path="./data/m4/dataset.hdf5",
        trunc=247,
        shorter=1000,
    )

def ett_long():
    return problem_spec_from_hdf5_standard(
        file_path="./data/ett_long/dataset.hdf5",
        trunc=2000,
    )

def ett_long_short():
    return problem_spec_from_hdf5_standard(
        file_path="./data/ett_long/dataset.hdf5",
        trunc=2000,
        shorter=1000,
    )


def cheetah_medium_dense_1000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/cheetah_medium_dense/dataset.hdf5",
        trunc=1000,
    )


def cheetah_medium_dense_500():
    return problem_spec_from_hdf5_standard(
        file_path="./data/cheetah_medium_dense/dataset.hdf5",
        trunc=500,
    )


def large_cheetah_medium_dense_1000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/cheetah_medium_dense/dataset_large.hdf5",
        trunc=1000,
    )

def large_cheetah_medium_dense_1000_shorter():
    return problem_spec_from_hdf5_standard(
        file_path="./data/cheetah_medium_dense/dataset_large.hdf5",
        trunc=1000,
        shorter=1000,
    )

def large_cheetah_medium_deter_1000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/cheetah_medium_deter/dataset_large.hdf5",
        trunc=1000,
    )

def large_cheetah_medium_dense_500():
    return problem_spec_from_hdf5_standard(
        file_path="./data/cheetah_medium_dense/dataset_large.hdf5",
        trunc=500,
    )


def hopper_medium_dense_1000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/hopper_medium_dense/dataset.hdf5",
        trunc=1000,
    )


def hopper_medium_dense_500():
    return problem_spec_from_hdf5_standard(
        file_path="./data/hopper_medium_dense/dataset.hdf5",
        trunc=500,
    )


def large_hopper_medium_dense_1000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/hopper_medium_dense/dataset_large.hdf5",
        trunc=1000,
    )

def large_hopper_medium_deter_1000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/hopper_medium_deter/dataset_large.hdf5",
        trunc=1000,
    )

def large_hopper_medium_dense_500():
    return problem_spec_from_hdf5_standard(
        file_path="./data/hopper_medium_dense/dataset_large.hdf5",
        trunc=500,
    )


def walker_medium_dense_1000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/walker_medium_dense/dataset.hdf5",
        trunc=1000,
    )


def walker_medium_dense_500():
    return problem_spec_from_hdf5_standard(
        file_path="./data/walker_medium_dense/dataset.hdf5",
        trunc=500,
    )


def large_walker_medium_dense_1000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/walker_medium_dense/dataset_large.hdf5",
        trunc=1000,
    )

def large_walker_medium_deter_1000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/walker_medium_deter/dataset_large.hdf5",
        trunc=1000,
    )

def large_walker_medium_dense_500():
    return problem_spec_from_hdf5_standard(
        file_path="./data/walker_medium_dense/dataset_large.hdf5",
        trunc=500,
    )


def ks100_500():
    return problem_spec_from_hdf5_standard(
        file_path="./data/ks_100_hard/dataset.hdf5",
        trunc=500,
    )


def ks100_1000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/ks_100_hard/dataset.hdf5",
        trunc=1000,
    )


def large_ks100_500():
    return problem_spec_from_hdf5_standard(
        file_path="./data/ks_100_hard/dataset_large.hdf5",
        trunc=500,
    )


def large_ks100_1000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/ks_100_hard/dataset_large.hdf5",
        trunc=1000,
    )

def large_turbulence_1000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/turbulence/dataset_large.hdf5",
        trunc=1000,
    )

def large_cahn_hillard_1000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/cahn_hillard/dataset_large.hdf5",
        trunc=1000,
    )

def electricity():
    return problem_spec_from_hdf5_standard(
        file_path="./data/electricity/dataset.hdf5",
        trunc=1440,
    )


def electricity_short():
    return problem_spec_from_hdf5_standard(
        file_path="./data/electricity/dataset.hdf5",
        trunc=1440,
        shorter=1000,
    )


def pems_bay():
    return problem_spec_from_hdf5_standard(
        file_path="./data/pems_bay/dataset.hdf5",
        trunc=288,
    )


def pems_bay_short():
    return problem_spec_from_hdf5_standard(
        file_path="./data/pems_bay/dataset.hdf5",
        trunc=288,
        shorter=1000,
    )


def m4_247():
    return problem_spec_from_hdf5_standard(
        file_path="./data/m4/dataset.hdf5",
        trunc=247,
    )


def tourism_228():
    return problem_spec_from_hdf5_standard(
        file_path="./data/tourism/dataset.hdf5",
        trunc=228,
    )

def tourism_short():
    return problem_spec_from_hdf5_standard(
        file_path="./data/tourism/dataset.hdf5",
        trunc=228,
        shorter=1000,
    )

def large_mackey_glass_2000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/mackey_glass/dataset_large.hdf5",
        trunc=2000,
    )

def large_mackey_glass_2000_shorter():
    return problem_spec_from_hdf5_standard(
        file_path="./data/mackey_glass/dataset_large.hdf5",
        trunc=2000,
        shorter=1000,
    )

def large_lorenz_2000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/lorenz/dataset_large.hdf5",
        trunc=2000,
    )

def large_mso_2000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/mso/dataset_large.hdf5",
        trunc=2000,
    )

def large_sspiral_2000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/sspiral/dataset_large.hdf5",
        trunc=2000,
    )

def large_lotka_2000():
    return problem_spec_from_hdf5_standard(
        file_path="./data/lotka/dataset_large.hdf5",
        trunc=2000,
    )
