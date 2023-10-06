# Unified Long-Horizon Time-Series Benchmark

## Usage
### Download data (do it first)
- `python3 src/download_data.py`
### run models
- `python3 src/train_py <configuration_name> <checkpoint_name> <noload|load_last|load_best> --problem_spec_name X --model_instance Y [<other model/workflow dependent arguments>]`

### CLI arguments:
- `configuration_name` - positional - name of the configuration, defined in src/configurations/configurations_ode.py
    - `fullconfig_N` - regular training for trajectories of length `N` without Curriculum Learning
    - `clconfig_N` - training for trajectories of length `N` with Curriculum Learning
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


## Run (example):
`python3 src/train.py fullconfig_247 test_1 noload --problem_spec_name m4_short --model_instance latent_ltsf_nlinear_96 --enc_dim 32 --latent_dim 2 --batch_size 64 --pretraining_len 100 --use_neptune 1`
