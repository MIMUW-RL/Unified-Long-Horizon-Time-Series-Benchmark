import random
from math import ceil
from time import time

import torch

import utils
from optimizer import EMAOptimizer
from utils import neptune_logging

from .base_trainer import BaseTrainer


class ODEINTWEventsTrainer(BaseTrainer):
    def train_run_middle(
        self,
        batch_size,
        ode_optimizer,
        schedulers,
        train_enc_dec,
        random_range,
        method,
        test=True,
        start_only=False,
    ):
        if random_range[1] - 1 == random_range[0]:
            random_seq_len = True
        else:
            random_seq_len = False

        self.model.train()

        t0 = time()
        train_tsdata = self.problem_spec.train_tsdata.shuffled(
            cap=self.trajectories_per_epoch
        )
        augumented_tsdata = train_tsdata.augumented(
            random_range[1] - 1, aug_step=self.aug_step
        )
        shuffled_tsdata = augumented_tsdata.shuffled(cap=200000)
        dataset_size = shuffled_tsdata.shape[0]
        train_batch_size = batch_size
        no_fits = ceil(dataset_size / train_batch_size)

        if train_enc_dec:
            if hasattr(self.model, "rec"):
                self.model.rec.set_freeze(False)
            if hasattr(self.model, "dec"):
                self.model.dec.set_freeze(False)
        else:
            if hasattr(self.model, "rec"):
                self.model.rec.set_freeze(True)
            if hasattr(self.model, "dec"):
                self.model.dec.set_freeze(True)

        epoch_stats = []
        for itr, sampled_tsdata in enumerate(
            shuffled_tsdata.generate_samples(int(train_batch_size))
        ):
            fit_id = itr + 1

            if random_seq_len:
                rand_seq_len = random.randint(random_range[0], random_range[1])
                sampled_tsdata = sampled_tsdata.truncated_by_obs(
                    rand_seq_len, None
                )

            stats = self.fit_step(
                tsdata=sampled_tsdata,
                metrics=self.problem_spec.metrics,
                optimizers=(ode_optimizer,),
                schedulers=schedulers,
                method=method,
            )
            print("fit {}/{}, {}".format(fit_id, no_fits, stats), end="\r")
            epoch_stats.append(stats.results)

        print()

        self.model.eval()

        mses = sum([results["mse"] for results in epoch_stats]) / len(
            epoch_stats
        )

        train_results = utils.model_results.ModelResults(
            {
                "train_mse": mses,
            }
        )

        self.save()

        t1 = time()
        train_set_results_hard = self.problem_spec.test_train(
            self,
            batch_size * 2,
            discrete_updates=True,
            method=method,
            data_len=self.test_train_data_len,
            atol=self.atol,
            rtol=self.rtol,
            min_dt=self.min_dt,
            sample_vae=self.sample_vae,
        )
        if test:
            test_set_results_hard = self.problem_spec.test(
                self,
                batch_size * 2,
                discrete_updates=True,
                method=method,
                data_len=self.test_data_len,
                atol=self.atol,
                rtol=self.rtol,
                min_dt=self.min_dt,
                sample_vae=self.sample_vae,
            )

            t2 = time()
            print("training time:", t1 - t0, "testing time:", t2 - t1)
            return (
                train_results,
                train_set_results_hard,
                train_set_results_hard,
                test_set_results_hard,
                test_set_results_hard,
            )
        else:
            t2 = time()
            print("training time:", t1 - t0, "testing time:", t2 - t1)
            return (
                train_results,
                train_set_results_hard,
                train_set_results_hard,
            )

    def fit(self):
        batch_size = self.batch_size
        ode_params = list(self.model.parameters())

        seq_len = self.train_seq_len

        if self.optimizer == "Adam":
            ode_optimizer = torch.optim.AdamW(
                params=ode_params,
                lr=self.max_lr,
                betas=self.betas,
            )
        elif self.optimizer == "Adamax":
            ode_optimizer = torch.optim.Adamax(
                params=ode_params, lr=self.max_lr
            )
        else:
            raise ValueError(f"unknown optimizer: {self.optimizer}")

        ode_optimizer = EMAOptimizer(
            ode_optimizer, utils.DEFAULT_DEVICE, decay=0.9999
        )
        seq_len_short = self.pretraining_len

        if self.increasing_seq_lens and seq_len_short is None:
            raise ValueError("--pretraining_len is not set")

        while self.increasing_seq_lens and seq_len_short < int(1.0 * seq_len):
            eps = (
                self.pretraining_epochs
                if seq_len_short == self.pretraining_len
                else self.increasing_seq_lens_epochs
            )
            print("eps:", eps)
            for epoch in range(0, eps):
                if seq_len_short == self.pretraining_len:
                    for g in ode_optimizer.param_groups:
                        g["lr"] = self.pretraining_ode_lr
                else:
                    for g in ode_optimizer.param_groups:
                        g["lr"] = self.max_lr

                print(f"small seq_len {seq_len_short} with su")
                print("epoch {}".format(epoch + 1))
                (
                    train_results,
                    train_set_results_hard,
                    train_set_results_soft,
                    test_set_results_hard,
                    test_set_results_soft,
                ) = self.train_run(
                    batch_size=batch_size,
                    random_range=(seq_len_short, seq_len_short + 1),
                    ode_optimizer=ode_optimizer,
                    schedulers=(),
                    train_enc_dec=True
                    if seq_len_short == self.pretraining_len
                    else self.train_enc_dec_after_pretraining,
                    test=True,
                    method=self.solver_method,
                )

                neptune_logging.log_w_prefix(
                    self.neptune_run, "train_short", train_results.results
                )
                neptune_logging.log_w_prefix(
                    self.neptune_run,
                    "train_hard",
                    train_set_results_hard.results,
                )
                neptune_logging.log_w_prefix(
                    self.neptune_run,
                    "train_soft",
                    train_set_results_soft.results,
                )
                neptune_logging.log_w_prefix(
                    self.neptune_run,
                    "test_hard",
                    test_set_results_hard.results,
                )
                neptune_logging.log_w_prefix(
                    self.neptune_run,
                    "test_soft",
                    test_set_results_soft.results,
                )

                print("train_results:", train_results)
                print("train_set_results_hard:", train_set_results_hard)
                print("train_set_results_soft:", train_set_results_soft)
                print("test_set_results_hard:", test_set_results_hard)
                print("test_set_results_soft:", test_set_results_soft)

            seq_len_short += self.seq_len_step

        for g in ode_optimizer.param_groups:
            g["lr"] = self.max_lr

        for epoch in range(1000000):
            print("full seq_len with su, seq_len: {}".format(seq_len))
            print("epoch {}".format(epoch + 1))
            (
                train_results,
                train_set_results_hard,
                train_set_results_soft,
                test_set_results_hard,
                test_set_results_soft,
            ) = self.train_run(
                random_range=(seq_len, seq_len + 1),
                batch_size=batch_size,
                ode_optimizer=ode_optimizer,
                train_enc_dec=True
                if self.train_enc_dec_after_pretraining
                else not self.increasing_seq_lens,
                schedulers=(),
                test=True,
                method=self.solver_method,
            )

            neptune_logging.log_w_prefix(
                self.neptune_run, "train_short", train_results.results
            )
            neptune_logging.log_w_prefix(
                self.neptune_run, "train_hard", train_set_results_hard.results
            )
            neptune_logging.log_w_prefix(
                self.neptune_run, "train_soft", train_set_results_soft.results
            )
            neptune_logging.log_w_prefix(
                self.neptune_run, "test_hard", test_set_results_hard.results
            )
            neptune_logging.log_w_prefix(
                self.neptune_run, "test_soft", test_set_results_soft.results
            )

            print("train_results:", train_results)
            print("train_set_results_hard:", train_set_results_hard)
            print("train_set_results_soft:", train_set_results_soft)
            print("test_set_results_hard:", test_set_results_hard)
            print("test_set_results_soft:", test_set_results_soft)
