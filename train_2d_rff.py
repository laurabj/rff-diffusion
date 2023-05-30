# Imports
import ml_collections.config_flags
import torch
import wandb
from absl import app, flags
from data import get_dataset
from models import NoiseScheduler
from rff import RandomFourierFeatures, ArcCos
from utils import flatten_nested_dict, update_config_dict
from pyro.contrib.gp.kernels import RBF, Exponential
import matplotlib.pyplot as plt

ml_collections.config_flags.DEFINE_config_file(
    "config",
    "configs/2d.py",
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

        # Get dataset (x- and y-coordinates)
        X = get_dataset(config.dataset, n=config.n_train) # shape = (N, 2)

        # Set limits of image window
        xmin, xmax = -6, 6
        ymin, ymax = -6, 6

        # We need to stack T noisy versions of the dataset to get (N, T, 2)
        N, T = X.shape[0], config.T

        # Utility class that contains functions to denoise, add noise, and beta schedule.
        noise_scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            beta_schedule=config.beta_schedule)

        # Sample T independent timesteps per datapoint.
        timesteps = torch.randint(low=0, high=noise_scheduler.num_timesteps, size=(N, T))

        # Set kernel
        if (config.kernel == "RBF"):
            kernel = RBF
        elif (config.kernel == "Exponential"):
            kernel = Exponential
        elif (config.kernel == "ArcCos"):
            kernel = ArcCos
        elif (config.kernel == "DSKN"):
            kernel = "DSKN"

        # Remove before submission!
        print(f'X.shape: {X.shape}')
        print(f'timesteps.shape: {timesteps.shape}')

        all_X = [] # concatenated X, (N * T, 2)
        all_Y = [] # concatenated targets Y,  (N * T, 2)
        all_timesteps = []  # concatenated timesteps (N * T, 1)

        # Loop for generating noisy samples
        for i in range(T):
            # Get noise from normal distribution (mean = 0, variance = 1)
            noise = torch.randn(size=X.shape)  # eps, (N, 2)
            all_Y.append(noise)
            # Given x_0 = X and eps=noise, calculate x_t (noisy)
            noisy = noise_scheduler.add_noise(x_start=X, x_noise=noise, timesteps=timesteps[:, i]) # (N, 2)
            all_X.append(noisy)  # (N * T, 2)
            all_timesteps.append(timesteps[:, i].unsqueeze(1).float())  # (N, T)

        # Concatenate all noisy samples
        X = torch.cat(all_X, dim=0)  # (N * T, 2)
        Y = torch.cat(all_Y, dim=0)  # (N * T, 2)
        # We normalise timesteps from [1, T] -> [0, 1]
        flat_timesteps = torch.cat(all_timesteps, dim=0).squeeze()  # (N * T, 1)

        print(f'all_noisy.shape: {X.shape}')
        print(f'flat_timesteps.shape: {flat_timesteps.shape}')
        print(f'all_noise.shape: {Y.shape}')

        # Initialising the RFF model class also solves the linear system exactly.
        model = RandomFourierFeatures(
            x=X,
            y=Y,
            t=flat_timesteps,
            num_features=config.num_features,
            emb_size=config.embedding_size,
            time_emb=config.time_embedding,
            input_emb=config.input_embedding,
            sin_cos=config.sin_cos,
            noise=1e-4,
            kernel=kernel,
            depth=config.depth,
            growth_factor=config.growth_factor,
            sigma0=config.sigma0
            )

        # Sampling time
        model.eval()

        # Sample (eval_batch_size) number of 2D points with random coordinates
        # sampled from a normal distribution (mean = 0, variance = 1)
        sample = torch.randn(config.eval_batch_size, 2)

        # timesteps is list from num_timesteps-1 to 0 (reverse process)
        timesteps = list(range(len(noise_scheduler)))[::-1]

        # Compute stepsize for when to save samples (for visualisation)
        stepsize = int(config.num_timesteps / config.save_n_samples)

        # Generate samples
        samples = noise_scheduler.sample(x_T=sample, timesteps=timesteps, model=model, stepsize=stepsize)

        # Save the images to wandb.
        images = []
        for i, sample in enumerate(samples):
            plt.figure(figsize=(10, 10))
            plt.scatter(sample[:, 0], sample[:, 1])
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            images.append(wandb.Image(plt))
        wandb.log({"samples": images})

if __name__ == "__main__":
    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
