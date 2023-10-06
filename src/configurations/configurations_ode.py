from problem_specs.instances import get_obs_dim

from . import model_instances  # noqa

def clconfig_1000(**kwargs):
    latent_dim = kwargs["latent_dim"]
    problem_spec_name = kwargs["problem_spec_name"]
    model_instance_name = kwargs["model_instance"]
    obs_dim = get_obs_dim(problem_spec_name)
    enc_dim = kwargs["enc_dim"]
    hidden_dim = kwargs["hidden_dim"]
    pretraining_len = kwargs["pretraining_len"]
    dec_width = kwargs["dec_width"]
    dec_depth = kwargs["dec_depth"]
    model = eval(
        f"model_instances.{model_instance_name}(\
        latent_dim={latent_dim},\
        obs_dim={obs_dim},\
        enc_dim={enc_dim},\
        hidden_dim={hidden_dim},\
        dec_depth={dec_depth},\
        dec_width={dec_width},\
    )"
    )
    batch_size = kwargs["batch_size"]
    max_lr = 0.001
    return {
        "problem": problem_spec_name,
        "trainer": "su_ode",
        "trainer_args": {
            "batch_size": batch_size,
            "max_lr": max_lr,
            "pretraining_ode_lr": max_lr,
            "epochs": 100,
            "step_size": 0.01 / 4,
            "min_dt": 0.01 / 32,
            "train_seq_len": 1000,
            "solver_method": "dopri5",
            "pretraining_epochs": 10,
            "increasing_seq_lens": True,
            "trajectories_per_epoch": 18000,
            "increasing_seq_lens_epochs": 1,
            "betas": (0.9, 0.999),
            "test_train_data_len": 2000,
            "test_data_len": 100000000,
            "entropy_loss": False,
            "atol": 1e-3,
            "rtol": 1e-2,
            "entropy_factor": 0.2,
            "seq_len_step": 50,
            "pretraining_len": pretraining_len,
            "train_enc_dec_after_pretraining": True,
            "aug_step": 100,
            "optimizer": "Adam",
            "add_kl": False,
            "sample_vae": False,
        },
        "latent_dim": latent_dim,
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "model": model,
        "T": 0.2,
    }

def clconfig_1000_lesstraj(**kwargs):
    cf = clconfig_1000(**kwargs)
    cf["trainer_args"]["trajectories_per_epoch"] = 3000
    cf["trainer_args"]["pretraining_epochs"] = 5
    return cf

def fullconfig_1000(**kwargs):
    cf = clconfig_1000(**kwargs)
    cf["trainer_args"]["pretraining_epochs"] = 0
    cf["trainer_args"]["increasing_seq_lens"] = False
    cf["trainer_args"]["train_enc_dec_after_pretraining"] = True
    return cf

def clconfig_2000(**kwargs):
    cf = clconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 2000
    cf["trainer_args"]["seq_len_step"] = 100
    cf["trainer_args"]["pretraining_epochs"] = 5
    return cf

def clconfig_500(**kwargs):
    cf = clconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 500
    return cf

def clconfig_288(**kwargs):
    cf = clconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 288
    return cf

def fullconfig_288(**kwargs):
    cf = fullconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 288
    return cf

def clconfig_200(**kwargs):
    cf = clconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 200
    cf["trainer_args"]["max_lr"] = 0.001
    return cf

def fullconfig_500(**kwargs):
    cf = fullconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 500
    return cf

def fullconfig_1440(**kwargs):
    cf = fullconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 1440
    return cf

def clconfig_1440(**kwargs):
    cf = clconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 1440
    return cf

def fullconfig_2000(**kwargs):
    cf = fullconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 2000
    return cf

def fullconfig_200(**kwargs):
    cf = fullconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 200
    return cf

def clconfig_192(**kwargs):
    cf = clconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 192
    return cf

def fullconfig_192(**kwargs):
    cf = fullconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 192
    return cf

def clconfig_247(**kwargs):
    cf = clconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 247
    return cf

def fullconfig_247(**kwargs):
    cf = fullconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 247
    return cf

def clconfig_228(**kwargs):
    cf = clconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 228
    return cf

def fullconfig_228(**kwargs):
    cf = fullconfig_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 228
    return cf