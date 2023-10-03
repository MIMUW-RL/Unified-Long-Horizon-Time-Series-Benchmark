import os

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

import utils
from model import losses
from utils import neptune_logging, count_parameters
from utils.model_results import ModelResults


class BaseTrainer(nn.Module):
    def __init__(
        self,
        problem_spec,
        model,
        problem_spec_name,
        train_seq_len,
        batch_size,
        adjoint,
        max_lr,
        trainer_name,
        train_dir,
        configuration,
        solver_method,
        pretraining_epochs,
        increasing_seq_lens,
        trajectories_per_epoch,
        betas,
        increasing_seq_lens_epochs,
        test_train_data_len,
        test_data_len,
        entropy_loss,
        atol,
        rtol,
        min_dt,
        entropy_factor,
        seq_len_step,
        pretraining_len,
        train_enc_dec_after_pretraining,
        pretraining_ode_lr,
        checkpoint_name,
        aug_step,
        optimizer,
        add_kl,
        sample_vae,
        device=utils.DEFAULT_DEVICE,
    ):
        super(BaseTrainer, self).__init__()

        if not add_kl and sample_vae:
            raise ValueError("cannot sample vae when add_kl is False")

        self.model = model
        self.train_seq_len = train_seq_len
        self.batch_size = batch_size
        self.adjoint = adjoint
        self.max_lr = max_lr
        self.problem_spec = problem_spec
        self.solver_method = solver_method
        self.problem_spec_name = problem_spec_name
        self.trainer_name = trainer_name
        self.device = device
        self.train_dir = train_dir
        self.pretraining_epochs = pretraining_epochs
        self.increasing_seq_lens = increasing_seq_lens
        self.trajectories_per_epoch = trajectories_per_epoch
        self.betas = betas
        self.increasing_seq_lens_epochs = increasing_seq_lens_epochs
        self.test_train_data_len = test_train_data_len
        self.test_data_len = test_data_len
        self.entropy_loss = entropy_loss
        self.atol = atol
        self.rtol = rtol
        self.min_dt = min_dt
        self.entropy_factor = entropy_factor
        self.seq_len_step = seq_len_step
        self.pretraining_len = pretraining_len
        self.train_enc_dec_after_pretraining = train_enc_dec_after_pretraining
        self.pretraining_ode_lr = pretraining_ode_lr
        self.aug_step = aug_step
        self.optimizer = optimizer
        self.add_kl = add_kl
        self.sample_vae = sample_vae
        self.encoded_len = model.rec._rec_len

        self.best_val_loss = 99999999999999.0
        self.last_val_loss = None

        self.relu = torch.nn.functional.relu

        self._run = None

        self.put_mock_best_last_checkpoints()

        if neptune_logging.USE_NEPTUNE:
            self.init_neptune(
                configuration,
                checkpoint_name,
            )

    def put_mock_best_last_checkpoints(self):
        # raise NotImplementedError
        pass

    def init_neptune(self, configuration, checkpoint_name):
        run = neptune_logging.get_neptune_run(
            dict(
                problem_spec=self.problem_spec_name,
                trainer=self.trainer_name,
                configuration=configuration,
                checkpoint_name=checkpoint_name,
                number_of_parameters=count_parameters(self.model)
            ),
            track_files_path="./models_cpth/{}".format(
                configuration["problem"],
            ),
        )

        self._run = run

    @property
    def neptune_run(self):
        return self._run

    def train_run(self, *args, **kwargs):
        (
            train_results,
            train_set_results_hard,
            train_set_results_soft,
            test_set_results_hard,
            test_set_results_soft,
        ) = self.train_run_middle(*args, **kwargs)
        self.last_val_loss = test_set_results_soft["mse"]
        self.train_run_post()
        return (
            train_results,
            train_set_results_hard,
            train_set_results_soft,
            test_set_results_hard,
            test_set_results_soft,
        )

    def train_run_middle(self, *args, **kwargs):
        raise NotImplementedError

    def train_run_post(self):
        if self.last_val_loss < self.best_val_loss:
            self.best_val_loss = self.last_val_loss
            self.save_best_checkpoint()

    def fit_step(
        self,
        tsdata,
        metrics,
        optimizers,
        schedulers,
        method,
    ):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        result = self.model(
            tsdata=tsdata,
            method=method,
            atol=self.atol,
            rtol=self.rtol,
            min_dt=self.min_dt,
            sample_vae=self.sample_vae,
            discrete_updates=False,
        )
        pred_x = result["pred_x"]
        z0_mean = result["z0_mean"]
        z0_std = result["z0_std"]
        if self.model.model_type in ["standard_node"]:
            steps = result["steps"]

        if self.add_kl:
            error = losses.elbo_loss(
                pred_x,
                tsdata.dataset,
                (
                    z0_mean,
                    z0_std,
                ),
                try_omit_first=self.encoded_len,
            )
        else:
            error = losses.mse(
                pred_x,
                tsdata.dataset,
                obs_flag=tsdata.obs_flag,
                try_omit_first=self.encoded_len,
            )

        criterion = error

        if torch.isnan(criterion):
            criterion_nan = True
        else:
            criterion_nan = False

        metrics = {
            metric.__name__: metric(
                pred_x,
                tsdata.dataset,
                obs_flag=tsdata.obs_flag,
                first=self.encoded_len,
            ).item()
            for metric in metrics
        }

        if self.model.model_type in ["standard_node"]:
            aux_metrics = dict(
                T_su=self.model.T,
                criterion=criterion.item(),
                steps=steps.sum(dim=1).mean().item(),
            )
        else:
            aux_metrics = dict()

        if self.add_kl:
            aux_metrics["kl_divergence"] = (
                losses.kl_divergence(z0_mean, z0_std).mean().item()
            )

        all_metrics = {
            k: v
            for k, v in (list(metrics.items()) + list(aux_metrics.items()))
        }

        if not criterion_nan:
            criterion.backward()

            clip_grad_norm_(self.parameters(), 1.0, "inf", False)

            for optimizer in optimizers:
                optimizer.step()

            for scheduler in schedulers:
                scheduler.step()

        return ModelResults(all_metrics)

    def fit():
        raise NotImplementedError()

    def load(self, load_checkpoint, quiet=False):
        if load_checkpoint == "noload":
            return
        if self.train_dir is not None:
            train_dir = self.train_dir
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
            init_state_dict = self.state_dict()
            try:
                if load_checkpoint == "load_best":
                    ckpt_path = os.path.join(train_dir, "best_ckpt.pth")
                    checkpoint = torch.load(ckpt_path, map_location="cpu")
                    self.load_state_dict(checkpoint["state_dict"])
                    if not quiet:
                        print("Loaded ckpt from {}".format(ckpt_path))
                elif load_checkpoint == "load_last":
                    ckpt_path = os.path.join(train_dir, "last_ckpt.pth")
                    checkpoint = torch.load(ckpt_path, map_location="cpu")
                    self.load_state_dict(checkpoint["state_dict"])
                    if not quiet:
                        print("Loaded ckpt from {}".format(ckpt_path))
                else:
                    raise ValueError(
                        f"invalid load_checkpoint option: {load_checkpoint}"
                    )
            except Exception as error:
                self.load_state_dict(init_state_dict)
                if not quiet:
                    print(error)
                    print("failed to load")

    def save(self):
        utils.save_model(
            name="last_ckpt.pth",
            train_dir=self.train_dir,
            state_dict=self.state_dict(),
        )

    def save_best_checkpoint(self):
        utils.save_model(
            name="best_ckpt.pth",
            train_dir=self.train_dir,
            state_dict=self.state_dict(),
        )
