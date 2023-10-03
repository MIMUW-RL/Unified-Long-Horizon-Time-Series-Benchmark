import copy
from math import ceil, floor

import numpy as np
import torch

import utils
from model import losses
from utils.model_results import ModelResults


def default_metrics():
    return [
        losses.mse,
        losses.wmape_loss,
        losses.mse_far,
        losses.mae,
        losses.mae_far,
    ]


class TSData:
    def __init__(
        self,
        dataset,
        timestamps,
        obs_flag,
        out_frac,
    ):
        self.dataset = dataset
        self.timestamps = timestamps
        self.obs_flag = obs_flag
        self.out_frac = out_frac

        self._check_init_validity()

    def _check_init_validity(self):
        assert torch.is_tensor(self.dataset)
        assert torch.is_tensor(self.timestamps)
        assert torch.is_tensor(self.obs_flag), f"obs_flag is {self.obs_flag}"

        assert len(self.dataset.shape) == 3
        assert len(self.timestamps.shape) == 2
        assert len(self.obs_flag.shape) == 2

        shp0 = self.dataset.shape[0]
        shp1 = self.dataset.shape[1]

        assert self.dataset.shape[0] == shp0
        assert self.timestamps.shape[0] == shp0
        assert self.obs_flag.shape[0] == shp0

        assert self.dataset.shape[1] == shp1
        assert self.timestamps.shape[1] == shp1
        assert self.obs_flag.shape[1] == shp1

        assert torch.all(self.timestamps[:, :-1] < self.timestamps[:, 1:])

    @property
    def ts_size(self):
        ts_max = torch.max(self.timestamps, dim=1)
        ts_min = torch.min(self.timestamps, dim=1)
        ret_val = (ts_max.values - ts_min.values) * 100
        return ret_val - 1

    def augumented(self, seq_len, aug_step):
        augumented_dataset = utils.augument_dataset(
            self.dataset, seq_len, aug_step
        )
        reshaped_timestamps = self.timestamps.view(
            self.timestamps.shape[0], self.timestamps.shape[1], 1
        )
        augumented_timestamps = utils.augument_dataset(
            reshaped_timestamps, seq_len, aug_step
        ).view(-1, seq_len)
        reshaped_obs_flag = self.obs_flag.view(
            self.obs_flag.shape[0], self.obs_flag.shape[1], 1
        )
        augumented_obs_flag = utils.augument_dataset(
            reshaped_obs_flag, seq_len, aug_step
        ).view(-1, seq_len)

        ret_val = TSData(
            dataset=augumented_dataset,
            timestamps=augumented_timestamps,
            obs_flag=augumented_obs_flag,
            out_frac=self.out_frac,
        )

        return ret_val

    def shuffled(self, cap=100000):
        idxs = np.arange(self.shape[0])
        np.random.shuffle(idxs)
        idxs = torch.tensor(idxs[:cap])

        shuffled_dataset = self.dataset[idxs]
        shuffled_timestamps = self.timestamps[idxs]
        shuffled_obs_flag = (
            None if self.obs_flag is None else self.obs_flag[idxs]
        )

        return TSData(
            dataset=shuffled_dataset,
            timestamps=shuffled_timestamps,
            obs_flag=shuffled_obs_flag,
            out_frac=self.out_frac,
        )

    @property
    def shape(self):
        return self.dataset.shape

    def generate_samples(self, batch_size):
        dataset_size = self.shape[0]
        no_fits = ceil(dataset_size / batch_size)
        batch_size = min(batch_size, dataset_size)
        for itr in range(no_fits):
            base = (itr % (dataset_size // batch_size)) * batch_size
            yielded_dataset = self.dataset[base : base + batch_size].to(
                utils.DEFAULT_DEVICE
            )
            yielded_timestamps = self.timestamps[base : base + batch_size].to(
                utils.DEFAULT_DEVICE
            )
            yielded_obs_flag = (
                None
                if self.obs_flag is None
                else self.obs_flag[base : base + batch_size].to(
                    utils.DEFAULT_DEVICE
                )
            )

            tsdata = TSData(
                dataset=yielded_dataset,
                timestamps=yielded_timestamps,
                obs_flag=yielded_obs_flag,
                out_frac=None,
            )

            if self.out_frac is None:
                yield tsdata
            else:
                raise NotImplementedError

    def truncated_by_obs(self, trunc_len, out_frac):
        """
        returns a TSData with trunc_len first observations
        """
        if trunc_len is None:
            return copy.copy(self)
        else:
            return TSData(
                dataset=self.dataset[:, :trunc_len],
                timestamps=self.timestamps[:, :trunc_len],
                obs_flag=None
                if self.obs_flag is None
                else self.obs_flag[:, :trunc_len],
                out_frac=out_frac,
            )

    def truncated_by_time(self, trunc_len):
        raise NotImplementedError

    def random_removed_at(self, random_start, random_end, out_frac, fill_na):
        np.random.seed(42)
        full_batch_size = self.shape[0]
        work_len = (
            self.shape[1]
            if random_start is None or random_end is None
            else random_end - random_start
        )
        random_start = 0 if random_start is None else random_start
        random_end = self.shape[1] if random_end is None else random_end
        randomized_idxs = np.arange(work_len * full_batch_size)
        np.random.shuffle(randomized_idxs)
        randomized_idxs = torch.tensor(
            randomized_idxs.reshape(full_batch_size, work_len)
        )
        randomized_idxs = torch.sort(randomized_idxs, dim=-1).indices
        out_idxs_len = floor(work_len * out_frac)
        randomized_idxs = randomized_idxs[:, out_idxs_len:]
        randomized_idxs = torch.sort(randomized_idxs).values + random_start

        first_idxs = np.arange(random_start * full_batch_size) % random_start
        first_idxs = torch.tensor(
            first_idxs.reshape(full_batch_size, random_start)
        )

        last_len = self.shape[1] - random_end
        last_idxs = np.arange(last_len * full_batch_size) % last_len
        last_idxs = (
            torch.tensor(last_idxs.reshape(full_batch_size, last_len))
            + random_end
        )

        final_idxs = torch.cat(
            [first_idxs, randomized_idxs, last_idxs], axis=1
        )

        new_dataset = self.dataset.clone() * 0.0 + fill_na
        new_obs_flag = self.dataset[:, :, 0].clone() * 0.0

        for i in range(final_idxs.shape[0]):
            new_dataset[i : i + 1, final_idxs[i]] = self.dataset[
                i : i + 1, final_idxs[i]
            ]
            new_obs_flag[i : i + 1, final_idxs[i]] = 1.0

        return TSData(
            dataset=new_dataset,
            timestamps=self.timestamps,
            obs_flag=new_obs_flag,
            out_frac=None,
        )


class ProblemSpec:
    def __init__(
        self,
        train_dataset,
        test_dataset,
        train_timestamps,
        test_timestamps,
        train_out_frac,
        train_obs_flag=None,
        test_obs_flag=None,
        trunc=None,
        metrics=default_metrics(),
        norm=True,
        less_training_trajectories=None,
    ):
        if train_obs_flag is None:
            train_obs_flag = torch.ones(
                train_timestamps.shape, dtype=torch.float32
            )
        if test_obs_flag is None:
            test_obs_flag = torch.ones(
                test_timestamps.shape, dtype=torch.float32
            )

        self.train_tsdata = TSData(
            train_dataset,
            train_timestamps,
            obs_flag=train_obs_flag,
            out_frac=train_out_frac,
        )
        self.test_tsdata = TSData(
            test_dataset,
            test_timestamps,
            obs_flag=test_obs_flag,
            out_frac=None,
        )
        self.metrics = metrics

        if norm:
            train_dataset_for_scaling = self.train_tsdata.dataset
            reshaped = train_dataset_for_scaling.reshape(
                -1, train_dataset_for_scaling.size(2)
            )

            mean = torch.mean(reshaped, dim=0)
            std = torch.std(reshaped, dim=0)

            self.mean = mean
            self.std = std

            def norm_data(data):
                n_data = data - self.mean
                n_data = n_data / self.std
                return n_data

            self.train_tsdata.dataset = norm_data(self.train_tsdata.dataset)
            self.test_tsdata.dataset = norm_data(self.test_tsdata.dataset)

        self.train_tsdata = self.train_tsdata.truncated_by_obs(
            trunc, train_out_frac
        )
        self.test_tsdata = self.test_tsdata.truncated_by_obs(trunc, None)

        if less_training_trajectories:
            self.train_tsdata = self.train_tsdata.shuffled(cap=less_training_trajectories)

    def random_removed_at(self, random_range, out_frac, fill_na):
        if random_range is None:
            random_start, random_end = None, None
        else:
            random_start, random_end = random_range

        raise Exception(
            f"check if random_start and random_end is right {random_start} - {random_end}"  # noqa
        )
        train_ = self.train_tsdata
        test_ = self.test_tsdata.random_removed_at(
            random_start, random_end, out_frac, fill_na
        )

        ps = ProblemSpec(
            train_dataset=train_.dataset,
            test_dataset=test_.dataset,
            train_timestamps=train_.timestamps,
            test_timestamps=test_.timestamps,
            train_obs_flag=train_.obs_flag,
            test_obs_flag=test_.obs_flag,
            metrics=self.metrics,
            train_out_frac=out_frac,
            norm=False,
        )

        return ps

    def test_a_dataset(
        self,
        trainer,
        batch_size,
        tsdata,
        discrete_updates,
        method,
        atol,
        rtol,
        min_dt,
        sample_vae,
    ):
        assert isinstance(tsdata, TSData)

        no_fits = 0
        computed_metrics_all = []
        steps = []

        len_processed = 0
        for sampled_tsdata in tsdata.generate_samples(batch_size):
            sample_len = sampled_tsdata.shape[0]
            len_processed += sample_len
            no_fits += 1
            result = trainer.model.predict(
                sampled_tsdata,
                discrete_updates=discrete_updates,
                method=method,
                atol=atol,
                rtol=rtol,
                min_dt=min_dt,
                sample_vae=sample_vae,
            )
            pred_x = result["pred_x"]
            if "steps" in result.keys():
                steps_ = result["steps"]
                steps.append(steps_)
            computed_metrics = [
                (
                    metric.__name__,
                    metric(
                        pred_x,
                        sampled_tsdata.dataset,
                        sampled_tsdata.obs_flag,
                        first=trainer.encoded_len,
                    ).item(),
                )
                for metric in self.all_metrics()
            ]
            computed_metrics_all.append(computed_metrics)

        computed_metrics_avg = []

        for i in range(len(computed_metrics_all[0])):
            name = computed_metrics_all[0][i][0]
            sum_ = 0
            for j in range(no_fits):
                sum_ += computed_metrics_all[j][i][1]

            avg = sum_ / no_fits
            computed_metrics_avg.append(
                (
                    name,
                    avg,
                )
            )

        model_results = ModelResults({k: v for k, v in computed_metrics_avg})

        return model_results

    def score_a_small_dataset(
        self,
        trainer,
        dataset,
        samp_ts,
    ):
        dataset = torch.from_numpy(dataset)
        dataset = dataset.to(utils.DEFAULT_DEVICE)
        pred_x, _ = trainer.model.predict(
            dataset,
            samp_ts,
        )
        return pred_x.detach().cpu().numpy()

    def all_metrics(self):
        return self.metrics

    def test(
        self,
        trainer,
        batch_size,
        discrete_updates,
        method,
        data_len,
        atol,
        rtol,
        min_dt,
        sample_vae,
    ):
        return self.test_a_dataset(
            trainer,
            batch_size,
            tsdata=self.test_tsdata.shuffled(cap=data_len),
            discrete_updates=discrete_updates,
            method=method,
            atol=atol,
            rtol=rtol,
            min_dt=min_dt,
            sample_vae=sample_vae,
        )

    def test_train(
        self,
        trainer,
        batch_size,
        discrete_updates,
        method,
        data_len,
        atol,
        rtol,
        min_dt,
        sample_vae,
    ):
        return self.test_a_dataset(
            trainer,
            batch_size,
            tsdata=self.train_tsdata.shuffled(cap=data_len),
            discrete_updates=discrete_updates,
            method=method,
            atol=atol,
            rtol=rtol,
            min_dt=min_dt,
            sample_vae=sample_vae,
        )
