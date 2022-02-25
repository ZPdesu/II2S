from yacs.config import CfgNode as CN

cfg = CN()

cfg.IN_W_SPACE =  True
cfg.L2_REGULARIZER = False
cfg.L2_WEIGHT = 0.001

cfg.SEED = 121212 # help='manual seed to use'
#?? parser.add_argument('-loss_str', type=str, default="None", help='Loss function to use')
cfg.EPS = 0 # 'Target for downscaling loss (L2)')
cfg.TILE_LATENT = False # help='Whether to forcibly tile the same latent 18 times')
cfg.OPT_NAME = 'adam' # help='Optimizer to use in projected gradient descent')
cfg.STEPS=1500 # help='Number of optimization steps')
cfg.LEARNING_RATE = 0.01 # help='Learning rate to use during optimization')
cfg.LR_SCHEDULE = 'fixed' # help='fixed, linear1cycledrop, linear1cycle')
cfg.SAVE_INTERMEDIATE = 0 #help='Whether to store and save intermediate HR and LR images during optimization')
cfg.LOG_FREQ = 50  # How often to show a summary on the console



cfg.STYLEGAN2 = CN()
cfg.STYLEGAN2.SIZE = 1024
cfg.STYLEGAN2.CKPT = "models/stylegan2-ffhq-config-f.pt"
cfg.STYLEGAN2.CHANNEL_MULTIPLIER = 2
cfg.STYLEGAN2.LATENT = 512
cfg.STYLEGAN2.N_MLP = 8

