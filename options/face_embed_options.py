from argparse import ArgumentParser
class FaceEmbedOptions(object):


    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):

        # StyleGAN2 setting
        self.parser.add_argument('--size', type=int, default=1024)
        self.parser.add_argument('--ckpt', type=str, default="pretrained_models/ffhq.pt")
        self.parser.add_argument('--channel_multiplier', type=int, default=2)
        self.parser.add_argument('--latent', type=int, default=512)
        self.parser.add_argument('--n_mlp', type=int, default=8)

        # loss options
        self.parser.add_argument('--percept_lambda', type=float, default=1.0, help='Perceptual loss multiplier factor')
        self.parser.add_argument('--l2_lambda', type=float, default=1.0, help='L2 loss multiplier factor')
        self.parser.add_argument('--p_norm_lambda', type=float, default=0.001, help='P-norm Regularizer multiplier factor')

        # arguments
        self.parser.add_argument('--device', type=str, default='cuda')
        self.parser.add_argument('--seed', type=int, default=None)
        self.parser.add_argument('--tile_latent', action='store_true', help='Whether to forcibly tile the same latent N times')
        self.parser.add_argument('--opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
        self.parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate to use during optimization')
        self.parser.add_argument('--lr_schedule', type=str, default='fixed', help='fixed, linear1cycledrop, linear1cycle')
        self.parser.add_argument('--steps', type=int, default=1300, help='Number of optimization steps')
        self.parser.add_argument('--save_intermediate', action='store_true',
                            help='Whether to store and save intermediate HR and LR images during optimization')
        self.parser.add_argument('--save_interval', type=int, default=300, help='Latent checkpoint interval')
        self.parser.add_argument('--verbose', action='store_true', help='Print loss information')