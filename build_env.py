from envs.quadrotor_env import QuadrotorEnv
from envs.unicycle_env import UnicycleEnv
from envs.vsa_env import VSAEnv

"""
This file includes a function that simply returns one of the two supported environments. 
"""

def build_env(args):
    """Build our custom gym environment."""

    if args.env_name == 'Unicycle':
        return UnicycleEnv()
    elif args.env_name == 'Quadrotor':
        return QuadrotorEnv()
    elif args.env_name == 'VSA':
        return VSAEnv()
    else:
        raise Exception('Env {} not supported!'.format(args.env_name))
