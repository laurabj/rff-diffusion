import os
import ml_collections.config_flags
import torch
import wandb
from absl import app, flags
from data import get_mnist_data
from models import NoiseScheduler
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from unet import Unet
from utils import flatten_nested_dict, plot_samples, update_config_dict
from rff import RandomFourierFeatures, ArcCos
from pyro.contrib.gp.kernels import RBF, Exponential
from conv_kernel import ConvSimple

ml_collections.config_flags.DEFINE_config_file(
    "config",
    "configs/mnist.py",
    "Training configuration.",
    lock_config=True,
)

FLAGS = flags.FLAGS


def main(config):
     # Weights and Biases initialisation begins. ####################################
    wandb_kwargs = {
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "config": flatten_nested_dict(config.to_dict()),
        "mode": "online" if config.wandb.log else "disabled",
        "settings": wandb.Settings(code_dir=config.wandb.code_dir),
    }
    with wandb.init(**wandb_kwargs) as run:
        computed_configs = {}
        update_config_dict(config, run, computed_configs)
        # Weights and Biases initialisation ends. ####################################

        device = "cpu" # if config.get('cpu', False) else "cuda"

        X = get_mnist_data(image_size=config.image_size, digit=config.digit, n=config.n_train, num_digits=config.num_digits)
        X = torch.as_tensor(X, dtype=torch.float32, device=device)
        print(f'X.shape: {X.shape}')

        # Adjust this
        xmin, xmax = -6, 6
        ymin, ymax = -6, 6

        outdir = f"exps/{config.experiment_name}"
        os.makedirs(outdir, exist_ok=True)
        imgdir = f"{outdir}/images"
        os.makedirs(imgdir, exist_ok=True)

        noise_scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            beta_schedule=config.beta_schedule,
            device=device,)

        if (config.kernel == "RBF"):
            kernel = RBF
        elif (config.kernel == "Exponential"):
            kernel = Exponential
        elif (config.kernel == "ArcCos"):
            kernel = ArcCos
        elif (config.kernel == "Conv"):
            kernel = ConvSimple
        elif (config.kernel == "DeepConv"):
            kernel = "DeepConv"

        N, K, T, d = X.shape[0], config.num_trajectories, config.num_timesteps, X.shape[1]

        all_X = []  # list to hold all noisy datapoints across all trajectories and timesteps
        all_Y = []  # list to hold all noise across all trajectories and timesteps
        all_timesteps = []  # list to hold all timesteps across all trajectories and timesteps

        for _ in range(K):  # iterate over trajectories
            # Noise to initialise trajectory.
            noise = torch.randn(X.shape).to(device)  # eps, (N, d)
            for t in range(T):
                noisy = noise_scheduler.add_noise(X, noise, torch.ones(N, dtype=torch.int).to(device) * t) # (N, d)

                all_Y.append(noise)
                all_X.append(noisy)
                all_timesteps.append(torch.ones(N, dtype=torch.int).to(device) * t)

        # Concatenate all noisy samples
        X = torch.cat(all_X, dim=0)  # (N * T * K, d)
        Y = torch.cat(all_Y, dim=0)  # (N * T * K, d)
        flat_timesteps = torch.cat(all_timesteps, dim=0).squeeze()  # (N * T * K, 1)

        print(f'all_noisy.shape: {X.shape}')
        print(f'flat_timesteps.shape: {flat_timesteps.shape}')
        print(f'all_noise.shape: {Y.shape}')

        model = RandomFourierFeatures(
            x=X.to(device),
            y=Y.to(device),
            t=flat_timesteps.to(device),
            num_features=config.num_features,
            emb_size=config.embedding_size,
            time_emb=config.time_embedding,
            input_emb=config.input_embedding,
            sin_cos=config.sin_cos,
            noise=1e-4,
            device=device,
            kernel=kernel,
            stride=config.stride,
            patch_size=config.patch_size,
            image_size=config.image_size
            )

        model.eval()
        #sample = torch.randn(config.eval_batch_size, config.image_size*config.image_size).to(device)
        sample = torch.randn([config.eval_batch_size] + [config.image_size*config.image_size]).to(device)
        timesteps = list(range(len(noise_scheduler)))[::-1]

        stepsize = int(config.num_timesteps / config.save_n_samples)

        samples = noise_scheduler.sample(sample, timesteps, model, stepsize, clip=True)
        print(len(samples))
        for i in range(len(samples)):
            samples[i] = samples[i].unflatten(1, (1, 28, 28))

        # Save the images to wandb.
        images = plot_samples(samples, dataset_name="mnist", savepath=imgdir)
        wandb.log({"samples": images})

if __name__ == "__main__":

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
