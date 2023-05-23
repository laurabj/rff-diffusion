import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.experiment_name = "mnist_rff"
    config.train_batch_size = 64
    config.eval_batch_size = 16  # Number of sampled images to generate.

    config.image_size = 28 #32
    config.num_epochs = 100
    config.save_images_every = 10
    config.save_n_samples = 10
    config.learning_rate = 1e-3
    config.num_timesteps = 10 #50
    config.beta_schedule = "linear" #["linear", "quadratic", "cosine"]
    config.clip_samples = False
    # UNet Configs
    config.in_channels = 1 # MNIST is grayscale
    config.base_dim = 64
    config.dim_mults = [2, 4] # [4, 8, 16, 16]
    config.input_shape = [1, 28, 28] #[1, 32, 32]
    config.embedding_size = 128
    config.time_embedding = "sinusoidal"  #["sinusoidal", "learnable", "linear", "zero"]
    config.input_embedding = "identity" #["sinusoidal", "learnable", "linear", "identity"]

    # RFF specific configs
    config.num_features = 1000
    config.kernel = "RBF"
    config.num_trajectories = 10
    config.sin_cos = False
    config.digit = 1
    config.num_digits = 10
    config.patch_size = 3
    config.stride = 1
    config.n_train = 1000
    config.depth = 1

    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = False
    config.wandb.project = "rff_diffusion"# "rff_diffusions"
    config.wandb.entity = "laurabjustesen" #"shreyaspadhy"
    config.wandb.code_dir = "/home/sp2058/rff_diffusions/rff_diffusions"


    return config
