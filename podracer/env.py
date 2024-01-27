import gymnasium as gym
from gymnasium.wrappers import TransformObservation
try:
    import envpool
except ImportError:
    envpool = None


def make_env(env_id, seed, num_envs):
    if envpool:
        envs = envpool.make(
            env_id,
            env_type="gymnasium",
            num_envs=num_envs,
            seed=seed,
        )
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
    else:
        envs = gym.make_vec(
            env_id,
            num_envs=num_envs,
            vector_kwargs=dict(shared_memory=False)
        )
    envs = TransformObservation(envs, lambda obs: obs['image'])
    envs.observation_space = envs.observation_space['image']
    envs.single_observation_space = envs.single_observation_space['image']
    return envs