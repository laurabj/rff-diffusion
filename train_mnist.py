
import os

import ml_collections.config_flags
import torch
import wandb
from absl import app, flags
from data import get_mnist_dataloaders
from models import NoiseScheduler
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from unet import Unet
from utils import flatten_nested_dict, plot_samples, update_config_dict

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
        device = "cpu" #if config.get('cpu', False) else "cuda"
        dataloader, _ = get_mnist_dataloaders(batch_size=config.train_batch_size, image_size=32)

        model = Unet(
            timesteps=config.num_timesteps,
            time_embedding_dim=config.embedding_size,
            in_channels=config.in_channels,
            out_channels=config.in_channels,
            base_dim=config.base_dim,
            dim_mults=config.dim_mults).to(device)

        noise_scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            beta_schedule=config.beta_schedule,
            device=device,)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=config.learning_rate,
            total_steps = config.num_epochs * len(dataloader),
            pct_start=0.25,
            anneal_strategy='cos')

        global_step = 0

        losses = []
        print("Training model...")
        outdir = f"exps/{config.experiment_name}"
        os.makedirs(outdir, exist_ok=True)

        print("Saving images...")
        imgdir = f"{outdir}/images"
        os.makedirs(imgdir, exist_ok=True)

        # Train the model using Gradient descent.
        progress_bar = tqdm(total=config.num_epochs)
        for epoch in tqdm(range(config.num_epochs)):
            model.train()

            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(dataloader):
                # (B, 2)
                batch, target = batch
                batch = batch.to(device)
                noise = torch.randn(batch.shape).to(device)  # Get eps noise.
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch.shape[0],)).long().to(device) # Get t

                # Get x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * eps
                noisy = noise_scheduler.add_noise(batch, noise, timesteps)

                # Get eps_{theta}(x_t, t)
                noise_pred = model(noisy, timesteps)
                # Minimise MSE loss between eps_{theta}(x_t, t) and eps
                loss = F.mse_loss(noise_pred, noise)
                loss.backward(loss)

                # Clip gradient norm of the model parameters to 1.0
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                logs = {"loss": loss.detach().item(), "step": global_step}
                wandb.log(logs)
                losses.append(loss.detach().item())
                global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            progress_bar.close()

            if epoch % config.save_images_every == 0 or epoch == config.num_epochs - 1:
                # generate data with the model to later visualize the learning process
                model.eval()

                # Somewhat hacky code to ensure sample is of size (eval_batch_size, *x_shape)
                sample = torch.randn([config.eval_batch_size] + config.input_shape).to(device)

                timesteps = list(range(len(noise_scheduler)))[::-1]

                stepsize = int(config.num_timesteps / config.save_n_samples)

                samples = noise_scheduler.sample(sample, timesteps, model, stepsize, clip=config.clip_samples)

                # Save the images to wandb.
                images = plot_samples(samples, dataset_name="mnist", savepath=imgdir)
                wandb.log({"samples": images})

        print("Saving model...")
        torch.save(model.state_dict(), f"{outdir}/model.pth")

if __name__ == "__main__":

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
