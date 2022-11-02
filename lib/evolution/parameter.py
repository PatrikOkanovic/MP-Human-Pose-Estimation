"""
Arguments and hyper-parameters used in dataset evolution.
"""
import argparse


def parse_arg_evolution():
    parser = argparse.ArgumentParser(description='evolve.py')
    # Hyper-parameters
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    # Number of generation to run
    parser.add_argument('-G', type=int, default=1)
    # Synthetize enough (E) data with a target ratio after G generations
    parser.add_argument('-E', type=bool, default=True)
    parser.add_argument('-T', type=float, default=2.5)  # the target ratio
    # Fraction
    parser.add_argument('-F', type=float, default=0.1)
    # Apply mutation on skeleton orientation
    parser.add_argument('-M', type=bool, default=True)
    # Apply mutation on bone vector length
    parser.add_argument('-MBL', type=bool, default=True)
    # The mutation rate for bone vector length
    parser.add_argument('-MBLR', type=float, default=0.5)
    # Mutation rate of changing local limb orientation
    parser.add_argument('-MRL', type=float, default=0.3)
    # Mutation rate of changing global skeleton orientation
    parser.add_argument('-MRG', type=float, default=0.1)
    # Standrd deviation of Guassian noise (in degrees) for local limb mutation
    parser.add_argument('-SDL', type=float, default=10.0)
    # Standrd deviation of Guassian noise for global orientation mutation
    parser.add_argument('-SDG', type=float, default=30.0)
    # Merge the synthetic data with the initial population
    parser.add_argument('-Mer', type=bool, default=True)
    # Apply the crossover operator
    parser.add_argument('-CV', type=bool, default=True)
    # Apply constraint to rule out invalid poses
    parser.add_argument('-C', type=bool, default=True)
    # Threshold for valid bone vector
    parser.add_argument('-Th', type=int, default=9)
    # Visualize the synthetic skeleton during exploring the data space
    # Save the intermediate synthetic data after each generation
    parser.add_argument('-I', type=bool, default=False)
    # File name for saving
    parser.add_argument('-SN', type=str, default='evolved_data')
    # Usage: split and save evolved dataset
    parser.add_argument('-split', type=bool, default=False)
    parser.add_argument('-split_ratio', type=float, default=0.9)
    opt = parser.parse_args()
    return opt