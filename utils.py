from collections.abc import MutableMapping

import matplotlib.pyplot as plt
import ml_collections
import torch
import torchvision
import wandb
from torchvision.utils import save_image


# Taken from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten_nested_dict(nested_dict, parent_key="", sep="."):
    items = []
    for name, cfg in nested_dict.items():
        new_key = parent_key + sep + name if parent_key else name
        if isinstance(cfg, MutableMapping):
            items.extend(flatten_nested_dict(cfg, new_key, sep=sep).items())
        else:
            items.append((new_key, cfg))

    return dict(items)


def update_config_dict(config_dict: ml_collections.ConfigDict, run, new_vals: dict):
    config_dict.unlock()
    config_dict.update_from_flattened_dict(run.config)
    config_dict.update_from_flattened_dict(new_vals)
    run.config.update(new_vals, allow_val_change=True)
    config_dict.lock()


#torchvision ema implementation
#https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def plot_samples(samples, xmin=-6., xmax=6., ymin=-6., ymax=6., dataset_name="2d", savepath=None):

    images = []
    if dataset_name == "2d":
        for i, sample in enumerate(samples):
            plt.figure(figsize=(10, 10))
            plt.scatter(sample[:, 0], sample[:, 1])
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            images.append(wandb.Image(plt))
    else:
        for i, sample in enumerate(samples):
            save_image(sample, fp=f"{savepath}/samples_{i}.png", normalize=True)
            grid = torchvision.utils.make_grid(sample)
            images.append(wandb.Image(grid))

    return images
