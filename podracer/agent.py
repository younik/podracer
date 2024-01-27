from collections import deque
import jax
import jax.numpy as jnp
import numpy as np
from podracer import env, structs, policy
import time
import queue


def rollout(
    key: jax.random.PRNGKey,
    args,
    rollout_queue,
    params_queue: queue.Queue,
    writer,
    seed,
):
    envs = env.make_env(
        args.env_id,
        seed,
        args.local_num_envs,
    )
    len_actor_device_ids = len(args.actor_device_ids)
    global_step = 0
    start_time = time.time()

    # put data in the last index
    episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)

    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    rollout_queue_put_time = deque(maxlen=10)
    actor_policy_version = 0
    next_obs, _ = envs.reset()
    next_ter = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)
    next_tru = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)

    for update in range(1, args.num_updates + 2):
        update_time_start = time.time()
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        d2h_time = 0
        env_send_time = 0
        # NOTE: `update != 2` is actually IMPORTANT — it allows us to start running policy collection
        # concurrently with the learning process. It also ensures the actor's policy version is only 1 step
        # behind the learner's policy version
        params_queue_get_time_start = time.time()
        if args.concurrency:
            if update != 2:
                params = params_queue.get()
                # NOTE: block here is important because otherwise this thread will call
                # the jitted `get_action_and_value` function that hangs until the params are ready.
                # This blocks the `get_action_and_value` function in other actor threads.
                # See https://excalidraw.com/#json=hSooeQL707gE5SWY8wOSS,GeaN1eb2r24PPi75a3n14Q for a visual explanation.
                params.network_params["params"]["Dense_0"][
                    "kernel"
                ].block_until_ready()  # TODO: check if params.block_until_ready() is enough
                actor_policy_version += 1
        else:
            params = params_queue.get()
            actor_policy_version += 1
        params_queue_get_time.append(time.time() - params_queue_get_time_start)
        rollout_time_start = time.time()
        storage = []
        for _ in range(0, args.num_steps):
            cached_next_obs = next_obs
            cached_next_ter = next_ter
            cached_next_tru = next_tru
            global_step += args.local_num_envs * args.num_actor_threads * len_actor_device_ids * args.world_size
            inference_time_start = time.time()
            action, logprob, value, key = policy.get_action_and_value(params, cached_next_obs, key)
            inference_time += time.time() - inference_time_start

            d2h_time_start = time.time()
            cpu_action = np.array(action)
            d2h_time += time.time() - d2h_time_start

            env_send_time_start = time.time()
            next_obs, next_reward, next_ter, next_tru, info = envs.step(cpu_action)
            env_send_time += time.time() - env_send_time_start
            storage_time_start = time.time()

            storage.append(
                structs.Transition(
                    obs=cached_next_obs,
                    terminations=cached_next_ter,
                    truncations=cached_next_tru,
                    actions=action,
                    logprobs=logprob,
                    values=value,
                    rewards=next_reward,
                )
            )
            episode_returns += next_reward
            dones = jnp.logical_or(next_ter, next_tru)
            # TODO: when multiepisodes, returned_episode_returns is overrided multiple times
            returned_episode_returns = np.where(dones, episode_returns, returned_episode_returns)
            episode_returns *= jnp.logical_not(dones)
            episode_lengths += 1
            returned_episode_lengths = np.where(dones, episode_lengths, returned_episode_lengths)
            episode_lengths *= jnp.logical_not(dones)
            storage_time += time.time() - storage_time_start
    
        rollout_time.append(time.time() - rollout_time_start)

        avg_episodic_return = np.mean(returned_episode_returns)
        sharded_storage = structs.Transition.make_sharded(storage, args.learner_devices)
        # next_obs, next_done are still in the host
        sharded_next_obs = jax.device_put_sharded(np.split(next_obs, len(args.learner_devices)), devices=args.learner_devices)
        sharded_next_ter = jax.device_put_sharded(np.split(next_ter, len(args.learner_devices)), devices=args.learner_devices)
        sharded_next_tru = jax.device_put_sharded(np.split(next_tru, len(args.learner_devices)), devices=args.learner_devices)
        payload = (
            global_step,
            actor_policy_version,
            update,
            sharded_storage,
            sharded_next_obs,
            sharded_next_ter,
            sharded_next_tru,
            np.mean(params_queue_get_time),
        )
        rollout_queue_put_time_start = time.time()
        rollout_queue.put(payload)
        rollout_queue_put_time.append(time.time() - rollout_queue_put_time_start)

        if writer is not None:
            if update % args.log_frequency == 0:
                print(
                    f"global_step={global_step}, avg_episodic_return={avg_episodic_return}, rollout_time={np.mean(rollout_time)}"
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("stats/rollout_time", np.mean(rollout_time), global_step)
                writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
                writer.add_scalar("charts/avg_episodic_length", np.mean(returned_episode_lengths), global_step)
                writer.add_scalar("stats/params_queue_get_time", np.mean(params_queue_get_time), global_step)
                writer.add_scalar("stats/env_recv_time", env_recv_time, global_step)
                writer.add_scalar("stats/inference_time", inference_time, global_step)
                writer.add_scalar("stats/storage_time", storage_time, global_step)
                writer.add_scalar("stats/d2h_time", d2h_time, global_step)
                writer.add_scalar("stats/env_send_time", env_send_time, global_step)
                writer.add_scalar("stats/rollout_queue_put_time", np.mean(rollout_queue_put_time), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar(
                    "charts/SPS_update",
                    int(
                        args.local_num_envs
                        * args.num_steps
                        * len_actor_device_ids
                        * args.num_actor_threads
                        * args.world_size
                        / (time.time() - update_time_start)
                    ),
                    global_step,
                )


