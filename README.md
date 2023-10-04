# Unified Long-Horizon Time-Series Benchmark
## results presented in the paper https://arxiv.org/abs/2309.15946 
## Usage
### Download data (do it first)
- `python3 src/download_data.py`
### run models
- `python3 src/train_py <configuration_name> <checkpoint_name> <noload|load_last|load_best> --problem_spec_name X --model_instance Y [<other model/workflow dependent arguments>]`

### CLI arguments:
- `configuration_name` - positional - name of the configuration, defined in src/configurations/configurations_ode.py
- `checkpoint_name` - positional - name of the checkpoint
- `noload|load_last|load_best` - start from a fresh checkpoint | start from the last checkpoint | start from the best checkpoint (validation MSE)
- `--problem_spec_name` - name of the problem spec (preprocessed dataset), defined in src/problem_specs/instances.py
- `--model_instance` - name of the model, defined in src/configurations/model_instances.py
- `--latent_dim` - model's latent dimension (if applicable)
- `--enc_dim` - encoder's dimension (if applicable)
- `--hidden_dim` - model's hidden dimension (if applicable)
- `--batch_size` - batch size of the workflow
- `--pretraining_len` - starting trajectory length for the curriculum learning
- `--dec_depth` - decoder's depth (if applicable)
- `--use_neptune` - 1 if use neptune.ai for monitoring

## Requirements:
- install project_auxiliary/requirements.txt


## Run (examples):

### we report all experiments with `timeout 8h`; (ws=CL)

1. `python3 src/train.py ode_full_247 test_1 noload --problem_spec_name m4_short --model_instance latent_ltsf_nlinear_96 --enc_dim 32 --latent_dim 2 --batch_size 64 --pretraining_len 100 --use_neptune 0`
2. `python3 src/train.py ode_full_1000 "hopperseed500" noload --latent_dim 50 --problem_spec_name large_hopper_medium_dense_1000 --model_instance "obs_ltsf_nlinear_500" --enc_dim 50 --hidden_dim 100 --batch_size 100 --use_neptune 0 --pretraining_len 100 --max_lr 0.00001`
3. `python3 src/train.py wsode_2000 "deeparwsmackey96" noload --latent_dim 8 --problem_spec_name large_mackey_glass_2000 --model_instance "lstm_3_96" --hidden_dim 128 --batch_size 100 --use_neptune 0 --pretraining_len 250`
4. `python3 src/train.py ode_full_1000 "spacetimech250" noload --hidden_dim 256 --problem_spec_name large_cahn_hillard_1000 --model_instance "spacetime_250" --batch_size 5 --use_neptune 0 --pretraining_len 100`
5. `python3 src/train.py ode_full_1000 "deeparvanillahopperdeter500" noload --latent_dim 50 --problem_spec_name large_hopper_medium_deter_1000 --model_instance "lstm_3_500" --hidden_dim 128 --batch_size 100 --use_neptune 0 --pretraining_len 550`
6. `python3 src/train.py ode_full_1000 "lstmwalkerdeter96" noload --latent_dim 50 --problem_spec_name large_walker_medium_deter_1000 --model_instance "dlstm_3_96" --enc_dim 50 --hidden_dim 100 --batch_size 100 --use_neptune 0 --pretraining_len 100`

## ML-Ops:
We implement support for neptune.ai ml-ops that can be used by setting `--use_neptune 1` and appropriately configuring https://github.com/MIMUW-RL/Unified-Long-Horizon-Time-Series-Benchmark/blob/main/src/utils/neptune_logging.py.
### example runs:
Per each run in the benchmark we store metrics at neptune.ai, i.e., https://app.neptune.ai/cyranka/bench/runs/BEN-1530. Complete benchmark runs available under request.
