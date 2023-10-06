def mlp_layers(in_dim, out_dim, width, depth):
    if depth < 0:
        raise ValueError("")
    elif depth == 0:
        return [
            (
                in_dim,
                out_dim,
            )
        ]
    else:
        layers = [
            (
                in_dim,
                width,
            )
        ]
        layers += [(width, width) for _ in range(depth - 1)]
        layers.append(
            (
                width,
                out_dim,
            )
        )

        return layers


def mlp(
    in_dim, out_dim, width, depth, activation, resnet_style, use_batchnorm
):
    return {
        "type": "mlp",
        "layers": mlp_layers(in_dim, out_dim, width, depth),
        "activation": activation,
        "use_batchnorm": use_batchnorm,
    }


def mlp_linear(latent_dim):
    return {
        "type": "linear",
        "latent_size": latent_dim,
    }


def mlp_const(latent_dim):
    return {
        "type": "const",
        "latent_size": latent_dim,
    }


def mlp_zero(latent_dim):
    return {
        "type": "zero",
        "latent_size": latent_dim,
    }


def rnn(latent_dim, width, depth):
    return {
        "type": "rnn",
        "latent_dim": latent_dim,
        "width": width,
        "depth": depth,
    }


def lstm(latent_dim, width, depth):
    return {
        "type": "lstm",
        "latent_dim": latent_dim,
        "width": width,
        "depth": depth,
    }


def ltsf_linear(latent_dim, lookback):
    return {
        "type": "ltsf_linear",
        "latent_dim": latent_dim,
        "lookback": lookback,
    }


def ltsf_nlinear(latent_dim, lookback):
    return {
        "type": "ltsf_nlinear",
        "latent_dim": latent_dim,
        "lookback": lookback,
    }


def ltsf_dlinear(latent_dim, lookback):
    return {
        "type": "ltsf_dlinear",
        "latent_dim": latent_dim,
        "lookback": lookback,
    }


def classifier(
    latent_dim, width, depth, activation, resnet_style, use_batchnorm
):
    return {
        "layers": mlp_layers(latent_dim, 1, width, depth),
        "activation": activation,
        "use_batchnorm": use_batchnorm,
    }


def wavenet_enc(filter_dim, latent_dim, obs_dim, depth, activation):
    return {
        "name": "WaveNetEnc",
        "filter_dim": filter_dim,
        "depth": depth,
        "activation": activation,
        "latent_dim": latent_dim,
        "obs_dim": obs_dim,
    }


def mult_wavenet_enc(
    filter_dim,
    latent_dim,
    obs_dim,
    depth,
    additional_convs_per_layer,
    activation,
    n_out,
    encoded_len,
):
    return {
        "name": "MultWaveNetEnc",
        "filter_dim": filter_dim,
        "depth": depth,
        "activation": activation,
        "latent_dim": latent_dim,
        "obs_dim": obs_dim,
        "n_out": n_out,
        "additional_convs_per_layer": additional_convs_per_layer,
        "encoded_len": encoded_len,
    }
    

def lstm_enc(
    hidden_dim,
    latent_dim,
    obs_dim,
    depth,
    n_out,
    encoded_len,
):
    return {
        "name": "LSTMEnc",
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "obs_dim": obs_dim,
        "depth": depth,
        "n_out": n_out,
        "encoded_len": encoded_len,
    }


def mult_id_enc(encoded_len):
    return {
        "name": "MultIdEnc",
        "encoded_len": encoded_len,
    }


def id_dec():
    return {
        "type": "IdDec",
    }
