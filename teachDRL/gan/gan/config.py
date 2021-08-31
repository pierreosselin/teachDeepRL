class WGanGPConfig:
    n_epochs = 200
    batch_size = 64
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    n_cpu = 8
    latent_dim = 100
    img_size = 32
    channels = 1
    n_critic = 5
    clip_value = 0.01
    sample_interval = 400
    model_dir = '../../data/models'
    maze_type = 'pacman'
