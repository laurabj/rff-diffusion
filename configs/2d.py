import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.experiment_name = "2d_mlp"
    config.dataset = 'circle' #["circle", "dino", "line", "moons"]
    config.n_train = 5000
    config.train_batch_size = 32
    config.eval_batch_size = 1000

    config.num_epochs = 200
    config.save_images_every = 10
    config.save_n_samples = 10
    config.learning_rate = 1e-3
    config.num_timesteps = 50
    config.beta_schedule = "linear" #["linear", "quadratic"]

    config.embedding_size = 128
    config.hidden_size = 128
    config.hidden_layers = 3
    config.time_embedding = "sinusoidal"  #["sinusoidal", "learnable", "linear", "zero"]
    config.input_embedding = "identity" #["sinusoidal", "learnable", "linear", "identity"]

    # RFF specific configs
    config.num_features = 1000
    config.kernel = "RBF" # RBF, Exponential, DSKN
    config.sin_cos = False
    config.depth = 1
    config.growth_factor = 1.0
    config.sigma0 = 2.0
    config.T = 50

    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = False
    config.wandb.project = "rff_diffusion"
    config.wandb.entity = "laurabjustesen"
    config.wandb.code_dir = "/home/sp2058/rff_diffusions/rff_diffusions"


    return config
