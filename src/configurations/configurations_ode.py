from problem_specs.instances import get_obs_dim

from . import model_instances  # noqa


def wsode_1000(**kwargs):
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

def wsode_1000_lesstraj(**kwargs):
    cf = wsode_1000(**kwargs)
    cf["trainer_args"]["trajectories_per_epoch"] = 3000
    cf["trainer_args"]["pretraining_epochs"] = 5
    return cf

def ode_full_1000(**kwargs):
    cf = wsode_1000(**kwargs)
    cf["trainer_args"]["pretraining_epochs"] = 0
    cf["trainer_args"]["increasing_seq_lens"] = False
    cf["trainer_args"]["train_enc_dec_after_pretraining"] = True
    return cf

def ode_full_1000_smallr(**kwargs):
    cf = wsode_1000(**kwargs)
    cf["trainer_args"]["max_lr"] = 0.00001
    cf["trainer_args"]["pretraining_epochs"] = 0
    cf["trainer_args"]["increasing_seq_lens"] = False
    cf["trainer_args"]["train_enc_dec_after_pretraining"] = True
    return cf

def wsode_2000(**kwargs):
    cf = wsode_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 2000
    cf["trainer_args"]["seq_len_step"] = 100
    cf["trainer_args"]["pretraining_epochs"] = 5
    return cf

def wsode_2000_lesstraj(**kwargs):
    cf = wsode_2000(**kwargs)
    cf["trainer_args"]["trajectories_per_epoch"] = 1000
    cf["trainer_args"]["pretraining_epochs"] = 2
    return cf

def wsode_500(**kwargs):
    cf = wsode_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 500
    return cf

def wsode_288(**kwargs):
    cf = wsode_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 288
    return cf

def ode_full_288(**kwargs):
    cf = ode_full_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 288
    return cf

def ode_full_288_less_train(**kwargs):
    cf = ode_full_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 288
    cf["trainer_args"]["trajectories_per_epoch"] = 100
    return cf

def ode_full_288_smalllr(**kwargs):
    cf = ode_full_288(**kwargs)
    cf["trainer_args"]["max_lr"] = 0.00001
    return cf

def wsode_500_lr001(**kwargs):
    cf = wsode_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 500
    cf["trainer_args"]["max_lr"] = 0.001
    return cf


def wsode_500_lr001_noretrain(**kwargs):
    cf = wsode_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 500
    cf["trainer_args"]["max_lr"] = 0.001
    cf["trainer_args"]["train_enc_dec_after_pretraining"] = False
    cf["trainer_args"]["pretraining_epochs"] = 50
    return cf


def wsode_200(**kwargs):
    cf = wsode_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 200
    cf["trainer_args"]["max_lr"] = 0.001
    return cf


def ode_full_500(**kwargs):
    cf = ode_full_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 500
    return cf

def ode_full_1440(**kwargs):
    cf = ode_full_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 1440
    return cf

def wsode_1440(**kwargs):
    cf = wsode_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 1440
    return cf

def ode_full_1440_smalllr(**kwargs):
    cf = ode_full_1440(**kwargs)
    cf['trainer_args']['max_lr'] = 0.0001
    return cf

def ode_full_2000(**kwargs):
    cf = ode_full_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 2000
    return cf

def ode_full_2000_smallr(**kwargs):
    cf = ode_full_2000(**kwargs)
    cf["trainer_args"]["max_lr"] = 0.0001
    return cf


def ode_full_200(**kwargs):
    cf = ode_full_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 200
    return cf


def wsode_192(**kwargs):
    cf = wsode_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 192
    return cf


def ode_full_192(**kwargs):
    cf = ode_full_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 192
    return cf


def wsode_247(**kwargs):
    cf = wsode_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 247
    return cf


def ode_full_247(**kwargs):
    cf = ode_full_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 247
    return cf

def ode_full_247_smalllr(**kwargs):
    cf = ode_full_247(**kwargs)
    cf['trainer_args']['max_lr'] = 0.0001
    return cf

def wsode_228(**kwargs):
    cf = wsode_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 228
    return cf


def ode_full_228(**kwargs):
    cf = ode_full_1000(**kwargs)
    cf["trainer_args"]["train_seq_len"] = 228
    return cf

def ode_full_228_smalllr(**kwargs):
    cf = ode_full_228(**kwargs)
    cf["trainer_args"]["max_lr"] = 0.0001
    return cf
