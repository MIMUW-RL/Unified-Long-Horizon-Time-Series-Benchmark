from datetime import datetime

from build_from_configuration import build_from_configuration
from utils import neptune_logging
from utils.parse_args import parse_args

import configurations  # noqa

args = parse_args()

neptune_logging.USE_NEPTUNE = args.use_neptune == 1

n1 = datetime.now()

configuration_fun = eval("configurations.{}".format(args.configuration_name))
configuration = configuration_fun(
    latent_dim=args.latent_dim,
    problem_spec_name=args.problem_spec_name,
    model_instance=args.model_instance,
    enc_dim=args.enc_dim,
    hidden_dim=args.hidden_dim,
    batch_size=args.batch_size,
    pretraining_len=args.pretraining_len,
    dec_depth=args.dec_depth,
    dec_width=args.dec_width,
)

print(configuration)

trainer = build_from_configuration(configuration, args.checkpoint_name)

trainer.load(args.load_checkpoint)

trainer.fit()

n2 = datetime.now()

print("finished in {}".format(n2 - n1))
