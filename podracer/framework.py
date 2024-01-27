import queue
import threading
import time
from collections import deque

import flax
import jax
import numpy as np
from flax.training.train_state import TrainState
from podracer import agent, learner
from podracer.args import args


def run(policy_params, optimizer, writer, key):
    agent_state = TrainState.create(
        apply_fn=None,
        params=policy_params,
        tx=optimizer
    )
    agent_state = flax.jax_utils.replicate(agent_state, devices=args.learner_devices)


    params_queues = []
    rollout_queues = []

    unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
    for d_idx, d_id in enumerate(args.actor_device_ids):
        local_devices = jax.local_devices()
        device_params = jax.device_put(unreplicated_params, local_devices[d_id])
        for thread_id in range(args.num_actor_threads):
            params_queues.append(queue.Queue(maxsize=1))
            rollout_queues.append(queue.Queue(maxsize=1))
            params_queues[-1].put(device_params)
            threading.Thread(
                target=agent.rollout,
                args=(
                    jax.device_put(key, local_devices[d_id]),
                    args,
                    rollout_queues[-1],
                    params_queues[-1],
                    writer if d_idx == 0 and thread_id == 0 else None,
                    args.seed + d_idx * args.num_actor_threads + thread_id,  # TODO: split
                ),
            ).start()

    learners_update = jax.pmap(
        learner.update,
        axis_name="local_devices",
        devices=args.global_learner_devices,
    )
    learner_keys = jax.device_put_replicated(key, args.learner_devices)
    rollout_queue_get_time = deque(maxlen=10)
    learner_policy_version = 0
    while learner_policy_version < args.num_updates:
        learner_policy_version += 1
        rollout_queue_get_time_start = time.time()
        sharded_storages = []
        sharded_next_obss = []
        sharded_next_ters = []
        sharded_next_trus = []
        for d_idx, d_id in enumerate(args.actor_device_ids):
            for thread_id in range(args.num_actor_threads):
                (
                    global_step,
                    actor_policy_version,
                    update,
                    sharded_storage,
                    sharded_next_obs,
                    sharded_next_ter,
                    sharded_next_tru,
                    avg_params_queue_get_time,
                ) = rollout_queues[d_idx * args.num_actor_threads + thread_id].get()
                sharded_storages.append(sharded_storage)
                sharded_next_obss.append(sharded_next_obs)
                sharded_next_ters.append(sharded_next_ter)
                sharded_next_trus.append(sharded_next_tru)

        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        training_time_start = time.time()
        agent_state, learner_keys, logs = learners_update(
            agent_state,
            sharded_storages,
            sharded_next_obss,
            sharded_next_ters,
            sharded_next_trus,
            learner_keys,
        )
        unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_params, jax.local_devices()[d_id])
            for thread_id in range(args.num_actor_threads):
                params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)
        
                # record rewards for plotting purposes
        if learner_policy_version % args.log_frequency == 0:
            writer.add_scalar("stats/rollout_queue_get_time", np.mean(rollout_queue_get_time), global_step)
            writer.add_scalar(
                "stats/rollout_params_queue_get_time_diff",
                np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
                global_step,
            )
            writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
            writer.add_scalar("stats/rollout_queue_size", rollout_queues[-1].qsize(), global_step)
            writer.add_scalar("stats/params_queue_size", params_queues[-1].qsize(), global_step)
            print(
                global_step,
                f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={learner_policy_version}, training time: {time.time() - training_time_start}s",
            )
            writer.add_scalar(
                "charts/learning_rate", agent_state.opt_state[2][1].hyperparams["learning_rate"][-1].item(), global_step
            )
            for key, value in logs.items():
                writer.add_scalar(f"learner/{key}", value.item(), global_step)