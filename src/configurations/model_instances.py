from .common import id_dec, mult_id_enc, mult_wavenet_enc, mlp, classifier, lstm_enc, mult_mlp_enc


def standard_node_1(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    assert hidden_dim is not None
    return {
        "type": "standard_node",
        # "enc": mult_wavenet_enc(enc_dim, latent_dim, obs_dim, 5, 0, "ELU", 1),
        "enc": lstm_enc(enc_dim, latent_dim, obs_dim, 2, 1, 32),
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            max(latent_dim, obs_dim) * 4,
            4,
            "ELU",
            True,
            False,
        ),
    }

# ===ablation definitions===

SMALLDEC2 = 200
ENCLAYERS = 4

#===lstm encoders latent ode===

def latentode_1_lstm(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "standard_node",
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            1,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
    }

def latentode_2_lstm(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "standard_node",
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            2,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
    }

def latentode_8_lstm(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "standard_node",
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            8,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
    }

def latentode_96_lstm(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "standard_node",
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            96,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
    }

def latentode_125_lstm(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "standard_node",
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            125,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
    }

def latentode_168_lstm(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "standard_node",
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            168,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
    }

def latentode_250_lstm(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "standard_node",
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            250,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
    }

def latentode_325_lstm(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "standard_node",
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            325,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
    }
    
def latentode_336_lstm(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "standard_node",
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            336,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
    }

def latentode_425_lstm(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "standard_node",
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            425,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
    }

def latentode_500_lstm(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "standard_node",
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            500,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
    }

def latentode_720_lstm(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "standard_node",
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            720,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
    }

def latentode_1000_lstm(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "standard_node",
        "latent_ode": mlp(
            latent_dim, latent_dim, hidden_dim, 3, "ELU", True, False
        ),
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            1000,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
    }

def obs_ltsf_nlinear_144(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(144),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 144,
    }
    

def obs_ltsf_nlinear_168(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(168),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 168,
    }

def obs_ltsf_nlinear_500(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(500),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 500,
    }

def obs_ltsf_nlinear_1(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(1),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 1,
    }

def obs_ltsf_nlinear_2(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(2),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 2,
    }

def obs_ltsf_nlinear_4(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(4),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 4,
    }


def obs_ltsf_nlinear_8(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(8),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 8,
    }

def obs_ltsf_nlinear_16(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(16),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 16,
    }

def obs_ltsf_nlinear_32(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(32),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 32,
    }

def obs_ltsf_nlinear_96(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(96),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 96,
    }

def obs_ltsf_nlinear_250(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(250),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 250,
    }

def obs_ltsf_nlinear_336(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(336),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 336,
    }

def obs_ltsf_nlinear_720(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(720),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 720,
    }

def obs_ltsf_nlinear_1000(obs_dim, enc_dim, **kwargs):
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": mult_id_enc(1000),
        "dec": id_dec(),
        "latent_dim": obs_dim,
        "N": 1000,
    }


def lstm_1(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 1,
        "separate_lstm_encoder": False,
    }


def lstm_2(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 2,
        "separate_lstm_encoder": True,
    }

def lstm_3_1(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": False,
        "encoded_len": 1,
    }

def lstm_3_2(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": False,
        "encoded_len": 2,
    }

def lstm_3_8(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": False,
        "encoded_len": 8,
    }

def lstm_3_96(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": False,
        "encoded_len": 96,
    }

def lstm_3_144(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": False,
        "encoded_len": 144,
    }

def lstm_3_168(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": False,
        "encoded_len": 168,
    }

def lstm_3_250(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": False,
        "encoded_len": 250,
    }

def lstm_3_336(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": False,
        "encoded_len": 336,
    }

def lstm_3_500(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": False,
        "encoded_len": 500,
    }

def lstm_3_720(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": False,
        "encoded_len": 720,
    }

def lstm_3_1000(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": False,
        "encoded_len": 1000,
    }

def lstm_4(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 4,
        "separate_lstm_encoder": False,
    }


def lstm_5(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 5,
        "separate_lstm_encoder": False,
    }


def lstm_6(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 6,
        "separate_lstm_encoder": False,
    }


def lstm_7(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 7,
        "separate_lstm_encoder": False,
    }


def dlstm_1(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 1,
        "separate_lstm_encoder": True,
        "encoded_len": 75,
    }


def dlstm_2(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 2,
        "separate_lstm_encoder": True,
    }


def dlstm_3(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": True,
        "encoded_len": 75,
    }

def dlstm_3_1(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": True,
        "encoded_len": 1,
    }

def dlstm_3_2(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": True,
        "encoded_len": 2,
    }

def dlstm_3_8(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": True,
        "encoded_len": 8,
    }

def dlstm_3_96(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": True,
        "encoded_len": 96,
    }

def dlstm_3_144(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": True,
        "encoded_len": 144,
    }

def dlstm_3_168(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": True,
        "encoded_len": 168,
    }

def dlstm_3_250(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": True,
        "encoded_len": 250,
    }

def dlstm_3_336(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": True,
        "encoded_len": 336,
    }

def dlstm_3_500(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": True,
        "encoded_len": 500,
    }

def dlstm_3_720(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": True,
        "encoded_len": 720,
    }

def dlstm_3_1000(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 3,
        "separate_lstm_encoder": True,
        "encoded_len": 1000,
    }

def dlstm_4(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 4,
        "separate_lstm_encoder": True,
    }


def dlstm_5(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 5,
        "separate_lstm_encoder": True,
    }


def dlstm_6(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 6,
        "separate_lstm_encoder": True,
    }


def dlstm_7(obs_dim, hidden_dim, **kwargs):
    assert obs_dim is not None
    assert hidden_dim is not None
    return {
        "type": "lstm",
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "depth": 7,
        "separate_lstm_encoder": True,
    }

### === latent LTSF begin ===

def latent_ltsf_nlinear_1(latent_dim, obs_dim, enc_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            1,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
        "latent_dim": latent_dim,
        "N": 1,
    }

def latent_ltsf_nlinear_2(latent_dim, obs_dim, enc_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            2,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
        "latent_dim": latent_dim,
        "N": 2,
    }

def latent_ltsf_nlinear_8(latent_dim, obs_dim, enc_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            8,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
        "latent_dim": latent_dim,
        "N": 8,
    }

def latent_ltsf_nlinear_96(latent_dim, obs_dim, enc_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            96,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
        "latent_dim": latent_dim,
        "N": 96,
    }

def latent_ltsf_nlinear_144(latent_dim, obs_dim, enc_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            144,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
        "latent_dim": latent_dim,
        "N": 144,
    }

def latent_ltsf_nlinear_168(latent_dim, obs_dim, enc_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            168,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
        "latent_dim": latent_dim,
        "N": 168,
    }

def latent_ltsf_nlinear_250(latent_dim, obs_dim, enc_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            250,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
        "latent_dim": latent_dim,
        "N": 250,
    }

def latent_ltsf_nlinear_336(latent_dim, obs_dim, enc_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            336,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
        "latent_dim": latent_dim,
        "N": 336,
    }

def latent_ltsf_nlinear_500(latent_dim, obs_dim, enc_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            500,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
        "latent_dim": latent_dim,
        "N": 500,
    }

def latent_ltsf_nlinear_720(latent_dim, obs_dim, enc_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            720,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
        "latent_dim": latent_dim,
        "N": 720,
    }

def latent_ltsf_nlinear_1000(latent_dim, obs_dim, enc_dim, **kwargs):
    assert latent_dim is not None
    assert obs_dim is not None
    assert enc_dim is not None
    return {
        "type": "latent_ltsf_nlinear",
        "enc": lstm_enc(
            enc_dim,
            latent_dim,
            obs_dim,
            2,
            1,
            1000,
        ),
        "dec": mlp(
            latent_dim,
            obs_dim,
            SMALLDEC2,
            ENCLAYERS,
            "ELU",
            True,
            False,
        ),
        "latent_dim": latent_dim,
        "N": 1000,
    }

### === latent LTSF end ===

def spacetime_test(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 75,
        "hidden_dim": hidden_dim,
        "encoder_layers": 2,
        "decoder_layers": 2,
    }

def spacetime_1(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 1,
        "hidden_dim": hidden_dim,
        "encoder_layers": 2,
        "decoder_layers": 2,
    }

def spacetime_2(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 2,
        "hidden_dim": hidden_dim,
        "encoder_layers": 2,
        "decoder_layers": 2,
    }

def spacetime_8(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 8,
        "hidden_dim": hidden_dim,
        "encoder_layers": 2,
        "decoder_layers": 2,
    }

def spacetime_96(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 96,
        "hidden_dim": hidden_dim,
        "encoder_layers": 2,
        "decoder_layers": 2,
    }

def spacetime_144(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 144,
        "hidden_dim": hidden_dim,
        "encoder_layers": 2,
        "decoder_layers": 2,
    }

def spacetime_168(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 168,
        "hidden_dim": hidden_dim,
        "encoder_layers": 2,
        "decoder_layers": 2,
    }

def spacetime_250(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 250,
        "hidden_dim": hidden_dim,
        "encoder_layers": 2,
        "decoder_layers": 2,
    }

def spacetime_336(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 336,
        "hidden_dim": hidden_dim,
        "encoder_layers": 2,
        "decoder_layers": 2,
    }

def spacetime_500(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 500,
        "hidden_dim": hidden_dim,
        "encoder_layers": 2,
        "decoder_layers": 2,
    }

def spacetime_720(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 720,
        "hidden_dim": hidden_dim,
        "encoder_layers": 2,
        "decoder_layers": 2,
    }

def spacetime_1000(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 1000,
        "hidden_dim": hidden_dim,
        "encoder_layers": 2,
        "decoder_layers": 2,
    }

def spacetime_500_3layer(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 500,
        "hidden_dim": hidden_dim,
        "encoder_layers": 3,
        "decoder_layers": 3,
    }

def spacetime_1000_3layer(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "spacetime",
        "obs_dim": obs_dim,
        "embedding_dim": hidden_dim,
        "encoder_dim": hidden_dim,
        "encoded_len": 1000,
        "hidden_dim": hidden_dim,
        "encoder_layers": 3,
        "decoder_layers": 3,
    }

def nhits_test(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "nhits",
        "obs_dim": obs_dim,
        "encoded_len": 96,
        "n_stacks": 3,
        "n_blocks": 3,
        "n_hidden_layers_in_block": 2,
        "hidden_dim": 256,
        "n_freq_downsample": [168, 24, 1],
    }

def nhits_1(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "nhits",
        "obs_dim": obs_dim,
        "encoded_len": 1,
        "n_stacks": 3,
        "n_blocks": 3,
        "n_hidden_layers_in_block": 2,
        "hidden_dim": 256,
        "n_freq_downsample": [168, 24, 1],
    }

def nhits_2(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "nhits",
        "obs_dim": obs_dim,
        "encoded_len": 2,
        "n_stacks": 3,
        "n_blocks": 3,
        "n_hidden_layers_in_block": 2,
        "hidden_dim": 256,
        "n_freq_downsample": [168, 24, 1],
    }

def nhits_8(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "nhits",
        "obs_dim": obs_dim,
        "encoded_len": 8,
        "n_stacks": 3,
        "n_blocks": 3,
        "n_hidden_layers_in_block": 2,
        "hidden_dim": 256,
        "n_freq_downsample": [168, 24, 1],
    }

def nhits_96(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "nhits",
        "obs_dim": obs_dim,
        "encoded_len": 96,
        "n_stacks": 3,
        "n_blocks": 3,
        "n_hidden_layers_in_block": 2,
        "hidden_dim": 128,
        "n_freq_downsample": [168, 24, 1],
    }

def nhits_144(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "nhits",
        "obs_dim": obs_dim,
        "encoded_len": 144,
        "n_stacks": 3,
        "n_blocks": 3,
        "n_hidden_layers_in_block": 2,
        "hidden_dim": 128,
        "n_freq_downsample": [168, 24, 1],
    }

def nhits_168(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "nhits",
        "obs_dim": obs_dim,
        "encoded_len": 168,
        "n_stacks": 3,
        "n_blocks": 3,
        "n_hidden_layers_in_block": 2,
        "hidden_dim": 128,
        "n_freq_downsample": [168, 24, 1],
    }

def nhits_250(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "nhits",
        "obs_dim": obs_dim,
        "encoded_len": 250,
        "n_stacks": 3,
        "n_blocks": 3,
        "n_hidden_layers_in_block": 2,
        "hidden_dim": 128,
        "n_freq_downsample": [168, 24, 1],
    }

def nhits_336(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "nhits",
        "obs_dim": obs_dim,
        "encoded_len": 336,
        "n_stacks": 3,
        "n_blocks": 3,
        "n_hidden_layers_in_block": 2,
        "hidden_dim": 128,
        "n_freq_downsample": [168, 24, 1],
    }

def nhits_500(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "nhits",
        "obs_dim": obs_dim,
        "encoded_len": 500,
        "n_stacks": 3,
        "n_blocks": 3,
        "n_hidden_layers_in_block": 2,
        "hidden_dim": 128,
        "n_freq_downsample": [168, 24, 1],
    }

def nhits_720(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "nhits",
        "obs_dim": obs_dim,
        "encoded_len": 720,
        "n_stacks": 3,
        "n_blocks": 3,
        "n_hidden_layers_in_block": 2,
        "hidden_dim": 128,
        "n_freq_downsample": [168, 24, 1],
    }

def nhits_1000(latent_dim, obs_dim, enc_dim, hidden_dim, **kwargs):
    return {
        "type": "nhits",
        "obs_dim": obs_dim,
        "encoded_len": 1000,
        "n_stacks": 3,
        "n_blocks": 3,
        "n_hidden_layers_in_block": 2,
        "hidden_dim": 128,
        "n_freq_downsample": [168, 24, 1],
    }