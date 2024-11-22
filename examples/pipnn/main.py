import argparse
from ml_collections import config_dict


def get_default_config():
    """Get the default hyperparameter configuration."""
    config = config_dict.ConfigDict()

    config.learning_rate = 2E-3
    config.batch_size = 128
    config.num_epochs = 500
    config.molecule = 'A4B'
    config.poly_degree = 3
    config.n_layers = 2
    config.n_neurons = 128
    config.ntr = 1000
    config.nval = 1000
    config.ntst = 2000
    config.l0 = 0.1
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configure training parameters.")

    # Adding arguments for each configuration option
    parser.add_argument('--workdir', type=str,
                        default=None, help='Working directory')
    parser.add_argument('--learning_rate', type=float,
                        default=2E-3, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=2000,
                        help='Number of epochs for training.')
    parser.add_argument('--molecule', type=str,
                        default='A4B', help='Type of molecule.')
    parser.add_argument('--poly_degree', type=int, default=3,
                        help='Degree of the polynomial.')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of layers in the neural network.')
    parser.add_argument('--n_neurons', type=int, default=128,
                        help='Number of neurons per layer.')
    parser.add_argument('--ntr', type=int, default=1000,
                        help='Number of training examples.')
    parser.add_argument('--nval', type=int, default=1000,
                        help='Number of validation examples.')
    parser.add_argument('--ntst', type=int, default=2000,
                        help='Number of test examples.')
    parser.add_argument('--l0', type=float, default=1.,
                        help='Relative weighting between Energy and Loss.')
    parser.add_argument('--grad_bool', type=bool, default=False,
                        help='If True training with Forces.')

    return parser.parse_args()


def merge_configs():
    import os
    config = get_default_config()
    args = parse_args()

    # Update the configuration with any non-None values from the command line arguments
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

    if args.workdir is None:
        workdir = os.getcwd()
    else:
        workdir = args.workdir

    return config, workdir, args.grad_bool


def main():
    config, workdir, bool_grad = merge_configs()
    print("Effective Configuration:")
    print(config)
    print(workdir)

    if bool_grad:
        from train_w_grad import train_and_evaluate
        train_and_evaluate(config, workdir)
    else:
        from train import train_and_evaluate
        train_and_evaluate(config, workdir)


if __name__ == "__main__":
    main()
