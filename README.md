# Unified Long-Horizon Time-Series Benchmark
## results presented in the paper https://arxiv.org/abs/2309.15946 
## Usage
### Download data (do it first)
- `python3 src/download_data.py`

The benchmark datasets will be downloaded to `data/` dir, e.g., `data/hopper_medium_dense/dataset_large.hdf5`, or download manually from https://drive.google.com/drive/folders/1IfAHpka3hu2kM4j6ebzAPnGpLpSMnTlf?usp=drive_link

### run models
- `python3 src/train_py <configuration_name> <checkpoint_name> <noload|load_last|load_best> --problem_spec_name X --model_instance Y [<other model/workflow dependent arguments>]`

### CLI arguments:
- `configuration_name` - positional - name of the configuration, defined in src/configurations/configurations_ode.py
    - `fullconfig_N*` - regular training for trajectories of length `N` without Curriculum Learning
    - `clconfig_N*` - training for trajectories of length `N` with Curriculum Learning
    - configs are defined in `src/configurations/configurations_ode.py`, various parameters unavailable through CLI can be accessed, and new configurations can be created there.
- `checkpoint_name` - positional - name of the checkpoint
- `noload|load_last|load_best` - start from a fresh checkpoint | start from the last checkpoint | start from the best checkpoint (validation MSE)
- `--problem_spec_name` - name of the problem spec (preprocessed dataset), defined in src/problem_specs/instances.py
- `--model_instance` - name of the model, defined in src/configurations/model_instances.py
- `--latent_dim` - model's latent dimension (if applicable)
- `--enc_dim` - encoder's dimension (if applicable)
- `--hidden_dim` - model's hidden dimension (if applicable)
- `--batch_size` - batch size of the workflow
- `--pretraining_len` - starting trajectory length for the curriculum learning (if CL is used)
- `--dec_depth` - decoder's depth (if applicable)
- `--use_neptune` - add optional argument --use_neptune 1 , if using neptune.ai for monitoring


## Requirements:
- install project_auxiliary/requirements.txt


## Run (examples):

1. `python3 src/train.py fullconfig_247 test_1 noload --problem_spec_name m4_short --model_instance latent_ltsf_nlinear_96 --enc_dim 32 --latent_dim 2 --batch_size 64`
2. `python3 src/train.py fullconfig_1000 "hopperseed500" noload --latent_dim 50 --problem_spec_name large_hopper_medium_dense_1000 --model_instance "obs_ltsf_nlinear_500" --enc_dim 50 --hidden_dim 100 --batch_size 100`
3. `python3 src/train.py clconfig_2000 "deeparwsmackey96" noload --latent_dim 8 --problem_spec_name large_mackey_glass_2000 --model_instance "lstm_3_96" --hidden_dim 128 --batch_size 100 --pretraining_len 100`
4. `python3 src/train.py fullconfig_1000 "spacetimech250" noload --hidden_dim 256 --problem_spec_name large_cahn_hillard_1000 --model_instance "spacetime_250" --batch_size 5`
5. `python3 src/train.py fullconfig_1000 "deeparvanillahopperdeter500" noload --latent_dim 50 --problem_spec_name large_hopper_medium_deter_1000 --model_instance "lstm_3_500" --hidden_dim 128 --batch_size 100`
6. `python3 src/train.py fullconfig_1000 "lstmwalkerdeter96" noload --latent_dim 50 --problem_spec_name large_walker_medium_deter_1000 --model_instance "dlstm_3_96" --enc_dim 50 --hidden_dim 100 --batch_size 100`

_Remark: All of the reported experiments in the paper were performed with uniform 8h cap, i.e., with `timeout 8h`;_

## Latent NLinear model (modified LTSF NLinear)

In the paper, we introduced the latent NLinear model, which is our improvement of the LTSF NLinear model https://github.com/cure-lab/LTSF-Linear. The latent NLinear model works on latent states encoded using an LSTM encoder rather than raw observations, allowing for, e.g., treating time series having large spatial dimensions. Our benchmark demonstrated that it outperforms the vanilla NLinear model and performs consistently well with a time series of diverse characteristics; see below for a barplot summarizing our results.
![Unified-Long-Horizon-Time-Series-Benchmark](https://github.com/MIMUW-RL/Unified-Long-Horizon-Time-Series-Benchmark/blob/main/img/benchmark.png)
Refer to https://drive.google.com/drive/folders/1S16bDiihE6xARB1uMN1AeWq5ByCqkTMI?usp=drive_link for precise benchmark results.

## DeepAR + Curriculum Learning

We also introduce a variant of DeepAR with curriculum learning (CL). This enhanced model is demonstrated to outperform the vanilla DeepAR in our benchmark (see the plot above).

## ML-Ops:

We implement support for neptune.ai ml-ops that can be used by adding `--use_neptune 1` and appropriately configuring https://github.com/MIMUW-RL/Unified-Long-Horizon-Time-Series-Benchmark/blob/main/src/utils/neptune_logging.py.

### Example reports:
Per each run in the benchmark we store metrics at Neptune.ai, i.e., https://app.neptune.ai/cyranka/bench/runs/BEN-1530. Complete benchmark runs available under request.
