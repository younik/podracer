import uuid
import random
import jax
import os
from podracer.args import args
from rich.pretty import pprint
from tensorboardX import SummaryWriter
import jax
import numpy as np
from podracer import policy, utils, optimizer, framework


def main():
    # Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
    # Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
    os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
    os.environ["TF_CUDNN DETERMINISTIC"] = "1"

    args.local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    assert (
        args.local_num_envs % len(args.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (
        int(args.local_num_envs / len(args.learner_device_ids)) * args.num_actor_threads % args.num_minibatches == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
    if args.distributed:
        jax.distributed.initialize(
            local_device_ids=range(len(args.learner_device_ids) + len(args.actor_device_ids)),
        )
        print(list(range(len(args.learner_device_ids) + len(args.actor_device_ids))))

    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    args.num_envs = args.local_num_envs * args.world_size * args.num_actor_threads * len(args.actor_device_ids)
    args.batch_size = args.local_batch_size * args.world_size
    args.minibatch_size = args.local_minibatch_size * args.world_size
    args.num_updates = args.total_timesteps // (args.local_batch_size * args.world_size)
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    args.learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    args.actor_devices = [local_devices[d_id] for d_id in args.actor_device_ids]
    args.global_learner_devices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    pprint(args.dict_repr())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{uuid.uuid4()}"
    if args.track and args.local_rank == 0:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=args.dict_repr(),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in args.dict_repr().items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, policy_key = jax.random.split(key)

    # env setup
    envs = utils.make_env(args.env_id, args.seed, args.local_num_envs)
    args.action_size = envs.single_action_space.n
    obs_sample = envs.single_observation_space.sample()
    policy_params = policy.PolicyNetwork(
        args.action_size, args.channels, args.hiddens
    ).init(policy_key, np.array([obs_sample]))
    envs.close()

    tx = optimizer.get(args)
    framework.run(policy_params, tx, writer, key)

    writer.close


if __name__ == "__main__":
    main()