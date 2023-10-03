import model
import trainers
import utils
from model import LTSF, FullModel, LSTMModel, SpaceTime
from model.components.mlp import MLP
from model.components.encoders import (
    RecognitionWaveNet,
    MLPEnc,
    MultIdEnc,
    LSTMEnc,
)
from model.components.decoders import IdDec
import problem_specs.instances as problem_spec_instances  # noqa
from model.solvers import ODEINTWEvents


def build_wavenet_enc(enc):
    return RecognitionWaveNet(
        latent_dim=enc["latent_dim"],
        obs_dim=enc["obs_dim"],
        nfilters=enc["filter_dim"],
        depth=enc["depth"],
        activation=enc["activation"],
        use_batchnorm=enc["use_batchnorm"],
    )


def build_mult_wavenet_enc(enc):
    return RecognitionWaveNet(
        latent_dim=enc["latent_dim"],
        obs_dim=enc["obs_dim"],
        nfilters=enc["filter_dim"],
        depth=enc["depth"],
        activation=enc["activation"],
        additional_convs_per_layer=enc["additional_convs_per_layer"],
        encoded_len=enc["encoded_len"],
    )


def build_mult_mlp_enc(enc):
    return MLPEnc(
        encoded_len=enc["encoded_len"],
        latent_dim=enc["latent_dim"],
        obs_dim=enc["obs_dim"],
        hidden_dim=enc["hidden_dim"],
        depth=enc["depth"],
        activation=enc["activation"],
    )


def build_mult_id_enc(enc):
    return MultIdEnc(enc["encoded_len"])


def build_lstm_enc(enc):
    return LSTMEnc(
        obs_dim=enc["obs_dim"],
        latent_dim=enc["latent_dim"],
        hidden_dim=enc["hidden_dim"],
        depth=enc["depth"],
        encoded_len=enc["encoded_len"],
    )


def build_enc(enc):
    if enc["name"] == "WaveNetEnc":
        return build_wavenet_enc(enc)
    elif enc["name"] == "MultWaveNetEnc":
        return build_mult_wavenet_enc(enc)
    elif enc["name"] == "MultMLPEnc":
        return build_mult_mlp_enc(enc)
    elif enc["name"] == "MultIdEnc":
        return build_mult_id_enc(enc)
    elif enc["name"] == "LSTMEnc":
        return build_lstm_enc(enc)
    else:
        raise NotImplementedError(enc)


def build_mlp(latent_ode):
    return MLP(
        layers=latent_ode["layers"],
        activation=latent_ode["activation"],
        use_batchnorm=latent_ode["use_batchnorm"],
        add_last=latent_ode.get("add_last", False),
    )


def build_lstm(latent_ode):
    return LSTMModel(
        latent_dim=latent_ode["latent_dim"],
        hidden_dim=latent_ode["width"],
        depth=latent_ode["depth"],
    )


def build_latent_ode(latent_ode):
    if latent_ode["type"] == "mlp":
        return build_mlp(latent_ode)
    elif latent_ode["type"] == "lstm":
        return build_lstm(latent_ode)
    else:
        raise NotImplementedError(latent_ode)


def build_id_dec(dec):
    return IdDec()


def build_dec(dec):
    if dec["type"] == "mlp":
        return build_mlp(dec)
    elif dec["type"] == "IdDec":
        return build_id_dec(dec)
    else:
        raise NotImplementedError(dec)


def build_standard_node_model(model):
    return {
        "type": "standard_node",
        "enc": build_enc(model["enc"]).to(utils.DEFAULT_DEVICE),
        "latent_ode": build_latent_ode(model["latent_ode"]).to(
            utils.DEFAULT_DEVICE
        ),
        "dec": build_dec(model["dec"]).to(utils.DEFAULT_DEVICE),
    }


def build_latent_ltsf_linear_model(model):
    return {
        "type": "latent_ltsf_linear",
        "enc": build_enc(model["enc"]).to(utils.DEFAULT_DEVICE),
        "dec": build_dec(model["dec"]).to(utils.DEFAULT_DEVICE),
        "latent_dim": model["latent_dim"],
    }


def build_latent_ltsf_nlinear_model(model):
    return {
        "type": "latent_ltsf_nlinear",
        "enc": build_enc(model["enc"]).to(utils.DEFAULT_DEVICE),
        "dec": build_dec(model["dec"]).to(utils.DEFAULT_DEVICE),
        "latent_dim": model["latent_dim"],
        "N": model["N"],
    }


def build_lstm_model(model):
    return model


def build_spacetime_model(model):
    return model


def build_nhits_model(model):
    return model


def build_model(model):
    if model["type"] == "standard_node":
        return build_standard_node_model(model)
    elif model["type"] == "latent_ltsf_linear":
        return build_latent_ltsf_linear_model(model)
    elif model["type"] == "latent_ltsf_nlinear":
        return build_latent_ltsf_nlinear_model(model)
    elif model["type"] == "lstm":
        return build_lstm_model(model)
    elif model["type"] == "spacetime":
        return build_spacetime_model(model)
    else:
        raise ValueError(f"unknown model type: {model}")


def build_from_configuration(configuration, checkpoint_name):
    model = build_model(
        configuration["model"],
    )
    problem_spec = eval(
        "problem_spec_instances.{}".format(configuration["problem"])
    )()
    trainer_class = trainers.ODEINTWEventsTrainer

    if model["type"] in ["standard_node", "standard_su_node"]:
        solver = ODEINTWEvents(
            default_step_size=configuration["trainer_args"]["step_size"]
        )
        full_model = FullModel(
            model_type=model["type"],
            obs_dim=configuration["obs_dim"],
            latent_dim=configuration["latent_dim"],
            rec=model["enc"],
            dec=model["dec"],
            latent_ode=model["latent_ode"],
            T=configuration.get("T"),
            solver=solver,
        )
    elif model["type"] == "latent_ltsf_linear":
        full_model = LTSF(
            rec=model["enc"],
            dec=model["dec"],
            latent_dim=model["latent_dim"],
            out_values=configuration["trainer_args"]["train_seq_len"] - model["N"],
            ltsf_type="linear",
            N=model["N"],
        )
    elif model["type"] == "latent_ltsf_nlinear":
        full_model = LTSF(
            rec=model["enc"],
            dec=model["dec"],
            latent_dim=model["latent_dim"],
            out_values=configuration["trainer_args"]["train_seq_len"] - model["N"],
            ltsf_type="nlinear",
            N=model["N"],
        )
    elif model["type"] == "lstm":
        full_model = LSTMModel(
            obs_dim=model["obs_dim"],
            hidden_dim=model["hidden_dim"],
            depth=model["depth"],
            separate_lstm_encoder=model["separate_lstm_encoder"],
            encoded_len=model["encoded_len"],
        )
    elif model["type"] == "spacetime":
        full_model = SpaceTime(
            obs_dim=model["obs_dim"],
            embedding_dim=model["embedding_dim"],
            encoder_dim=model["encoder_dim"],
            encoded_len=model["encoded_len"],
            pred_len=configuration["trainer_args"]["train_seq_len"] - model["encoded_len"],
            hidden_dim=model["hidden_dim"],
            encoder_layers=model["encoder_layers"],
            decoder_layers=model["decoder_layers"],
        )
    else:
        raise ValueError(f"unknown model type: {model['type']}")

    full_model = full_model.to(utils.DEFAULT_DEVICE)

    count_parameters = 0

    for param in full_model.parameters():
        count_parameters += param.contiguous().view(-1).shape[0]

    print(f"\nModel has {count_parameters} parameters\n")

    trainer = trainer_class(
        problem_spec=problem_spec,
        model=full_model,
        problem_spec_name=configuration["problem"],
        adjoint=configuration["trainer_args"].get("adjoint"),
        train_seq_len=configuration["trainer_args"]["train_seq_len"],
        batch_size=configuration["trainer_args"]["batch_size"],
        max_lr=configuration["trainer_args"]["max_lr"],
        trainer_name="su_ode",
        train_dir="./models_cpth/{}/{}".format(
            configuration["problem"],
            checkpoint_name,
        ),
        configuration=configuration,
        solver_method=configuration["trainer_args"]["solver_method"],
        pretraining_epochs=configuration["trainer_args"]["pretraining_epochs"],
        increasing_seq_lens=configuration["trainer_args"][
            "increasing_seq_lens"
        ],
        trajectories_per_epoch=configuration["trainer_args"][
            "trajectories_per_epoch"
        ],
        betas=configuration["trainer_args"]["betas"],
        increasing_seq_lens_epochs=configuration["trainer_args"][
            "increasing_seq_lens_epochs"
        ],
        test_train_data_len=configuration["trainer_args"][
            "test_train_data_len"
        ],
        test_data_len=configuration["trainer_args"]["test_data_len"],
        entropy_loss=configuration["trainer_args"]["entropy_loss"],
        atol=configuration["trainer_args"]["atol"],
        rtol=configuration["trainer_args"]["rtol"],
        min_dt=configuration["trainer_args"]["min_dt"],
        entropy_factor=configuration["trainer_args"]["entropy_factor"],
        seq_len_step=configuration["trainer_args"]["seq_len_step"],
        pretraining_len=configuration["trainer_args"]["pretraining_len"],
        train_enc_dec_after_pretraining=configuration["trainer_args"][
            "train_enc_dec_after_pretraining"
        ],
        pretraining_ode_lr=configuration["trainer_args"]["pretraining_ode_lr"],
        checkpoint_name=checkpoint_name,
        aug_step=configuration["trainer_args"]["aug_step"],
        optimizer=configuration["trainer_args"]["optimizer"],
        sample_vae=configuration["trainer_args"]["sample_vae"],
        add_kl=configuration["trainer_args"]["add_kl"],
    )

    return trainer
