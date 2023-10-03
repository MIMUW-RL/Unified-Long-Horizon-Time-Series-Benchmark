import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configuration_name", type=str, help="configuration name"
    )
    parser.add_argument("checkpoint_name", type=str, help="checkpoint name")
    parser.add_argument(
        "load_checkpoint", type=str, help="noload|load_best|load_last"
    )
    parser.add_argument(
        "--use_neptune", type=bool, default=False, help="use neptune logging"
    )

    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--problem_spec_name", type=str, default=None)
    parser.add_argument("--model_instance", type=str, default=None)
    parser.add_argument("--enc_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--pretraining_len", type=int, default=None)
    parser.add_argument("--dec_depth", type=int, default=None)
    parser.add_argument("--dec_width", type=int, default=None)

    args = parser.parse_args()

    assert args.problem_spec_name is not None
    assert args.model_instance is not None
    assert args.batch_size is not None

    print("Cuda available:", torch.cuda.is_available())
    print("configuration name:", args.configuration_name)
    print("checkpoint_name:", args.checkpoint_name)
    print("load_checkpoint:", args.load_checkpoint)
    print("use neptune:", args.use_neptune)
    print("problem_spec_name:", args.problem_spec_name)
    print("latent_dim:", args.latent_dim)
    print("model_instance:", args.model_instance)

    return args
