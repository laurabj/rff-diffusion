
import os

import matplotlib.pyplot as plt
import ml_collections.config_flags
import torch
import wandb
from absl import app, flags
from data import get_dataset
from models import MLP, NoiseScheduler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import flatten_nested_dict, update_config_dict

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
        
        X = get_dataset(config.dataset, n=config.n_train)
        dataloader = DataLoader(
            X, batch_size=config.train_batch_size, shuffle=True, drop_last=True)
        xmin, xmax = -6, 6
        ymin, ymax = -6, 6

        model = MLP(
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            emb_size=config.embedding_size,
            time_emb=config.time_embedding,
            input_emb=config.input_embedding)

        noise_scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            beta_schedule=config.beta_schedule)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )
        
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
                noise = torch.randn(batch.shape)  # Get eps noise.
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch.shape[0],)).long() # Get t

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
                sample = torch.randn(config.eval_batch_size, 2)
                timesteps = list(range(len(noise_scheduler)))[::-1]

                stepsize = int(config.num_timesteps / config.save_n_samples)
                
                samples = noise_scheduler.sample(sample, timesteps, model, stepsize)

                print(len(samples))
                # Save the images to wandb.
                images = []
                for i, sample in enumerate(samples):
                    plt.figure(figsize=(10, 10))
                    plt.scatter(sample[:, 0], sample[:, 1])
                    plt.xlim(xmin, xmax)
                    plt.ylim(ymin, ymax)
                    images.append(wandb.Image(plt))
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