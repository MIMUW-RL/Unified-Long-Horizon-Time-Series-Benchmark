import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML


def count_parameters(model):
    count_parameters = 0

    for param in model.parameters():
        count_parameters += param.contiguous().view(-1).shape[0]

    return count_parameters


class RunningAverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.avg = 0
        self.first = True

    def update(self, val):
        if self.first:
            self.avg = val
            self.first = False
        self.avg = self.avg * self.momentum + val * (1 - self.momentum)


def max_pool_im(ar, div):
    height = ar.shape[-3]
    width = ar.shape[-2]

    new_shape = list(ar.shape)
    new_shape[-3] = height // div
    new_shape[-2] = width // div
    new_shape[-1] = 1
    new_ar = np.zeros(new_shape)

    for i in range(height // div):
        for j in range(width // div):
            avg_slice = ar[
                :, :, i * div : i * div + div, j * div : j * div + div, :
            ]
            avg_slice = np.max(avg_slice, axis=2)
            avg_slice = np.max(avg_slice, axis=2)
            avg_slice = np.max(avg_slice, axis=2)

            new_ar[:, :, i, j, 0] = avg_slice

    return new_ar.astype("float32")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    # elif torch.backends.mps.is_available():
        # return torch.device("mps")
    else:
        return torch.device("cpu")


DEFAULT_DEVICE = get_device()
DEFAULT_MISSING_VALUE = 0.0


def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use(
        "Agg"
    )  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=frames,
        interval=interval,
        blit=True,
        repeat=False,
    )
    return HTML(anim.to_html5_video())


def get_ts(start, seq_len, device=DEFAULT_DEVICE):
    return torch.from_numpy(
        np.linspace(start, start + (seq_len - 1) / 100, seq_len).astype(
            "float32"
        )
    ).to(device)


def transform(ar, mean, scale):
    return (ar - mean) / scale


def inverse_transform(ar, mean, scale):
    return (ar * scale) + mean


def augument_dataset(orig_dataset, seq_len, aug_step):
    assert torch.is_tensor(orig_dataset)
    assert len(orig_dataset.shape) == 3
    assert type(seq_len) == int

    full_len = orig_dataset.shape[1]
    o_len = orig_dataset.shape[0]
    dataset_size = 1

    while dataset_size * aug_step + seq_len < full_len:
        dataset_size += 1

    dataset_size *= o_len
    augumented_dataset = torch.zeros(
        (
            dataset_size,
            seq_len,
            orig_dataset.shape[2],
        )
    )

    i = 0

    while i * aug_step + seq_len < full_len:
        augumented_dataset[i * o_len : (i + 1) * o_len] = orig_dataset[
            :, i * aug_step : i * aug_step + seq_len
        ]
        i += 1

    end_dataset = orig_dataset[:, -seq_len:]
    augumented_dataset[-o_len:] = end_dataset

    return augumented_dataset


def save_model(name, train_dir, state_dict):
    if train_dir is not None:
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        ckpt_path = os.path.join(train_dir, name)

        torch.save(
            {
                "state_dict": state_dict,
            },
            ckpt_path,
        )


def load_model():
    raise NotImplementedError
