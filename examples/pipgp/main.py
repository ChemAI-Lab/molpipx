import argparse
from ml_collections import config_dict

def get_default_config():
    """Get the default hyperparameter configuration."""
    config = config_dict.ConfigDict()

    config.decay_steps = 700
    config.init_value = 0.0
    config.peak_value = 0.01
    config.warmup_steps = 75
    config.end_value = 0.0
    config.molecule = 'A4B'
    config.poly_degree = 3
    config.num_iter = 800
    config.ntr = 1000
    config.ntst = 5000
    config.kernel_type = 'Matern52'
    config.trainable_l = False

    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Configure training parameters.")

    # Adding arguments for each configuration option
    parser.add_argument('--workdir', type=str, default=None, help='Working directory.')
    parser.add_argument('--kernel_type', type=str, default='Matern52', help='The type of kernel to use in the Gaussian Process.')
    parser.add_argument('--decay_steps', type=int, default=700, help=' Positive integer, the total length of the schedule.')
    parser.add_argument('--peak_value', type=float, default=0.01, help='Peak value for scalar to be annealed at end of warmup.')
    parser.add_argument('--warmup_steps', type=int, default=75, help='Positive integer, the length of the linear warmup.')
    parser.add_argument('--end_value', type=float, default=0.0, help='End value of the scalar to be annealed.')
    parser.add_argument('--trainable_l', type=bool, default=False, help=' trainable morse variables length scale parameter.')
    parser.add_argument('--molecule', type=str, default='A4B', help='Type of molecule.')
    parser.add_argument('--poly_degree', type=int, default=3, help='Degree of the polynomial.')
    parser.add_argument('--ntr', type=int, default=1000, help='Number of training examples.')
    parser.add_argument('--ntst', type=int, default=5000, help='Number of test examples.')
    parser.add_argument('--num_iter', type=int, default=800, help='Number of iterations.')
    

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

    return config, workdir

def main():
    config, workdir = merge_configs()
    print("Effective Configuration:")
    print(config)
    print(workdir)

    from train import train_and_evaluate
    train_and_evaluate(config, workdir)


if __name__ == "__main__":
    main()
