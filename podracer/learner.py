
from typing import List
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from podracer import policy, structs
from podracer.args import args


@jax.jit
def update(
    agent_state: TrainState,
    sharded_storages: List,
    sharded_next_obs: List,
    sharded_next_ter: List,
    sharded_next_tru: List,
    key: jax.random.PRNGKey,
):
    storage = jax.tree_map(lambda *x: jnp.hstack(x), *sharded_storages)
    next_obs = jnp.concatenate(sharded_next_obs)
    next_ter = jnp.concatenate(sharded_next_ter)
    next_tru = jnp.concatenate(sharded_next_tru)
    next_done = jnp.logical_or(next_ter, next_tru)
    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    advantages, target_values = compute_gae(agent_state, next_obs, next_done, storage)
    if args.norm_adv: # NOTE: per-minibatch advantages normalization
        advantages = advantages.reshape(advantages.shape[0], args.num_minibatches, -1)
        advantages = (advantages - advantages.mean((0, -1), keepdims=True)) / (advantages.std((0, -1), keepdims=True) + 1e-8)
        advantages = advantages.reshape(advantages.shape[0], -1)

    def update_epoch(carry, _):
        agent_state, key = carry
        key, subkey = jax.random.split(key)

        def flatten(x):
            return x.reshape((-1,) + x.shape[2:])

        # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(subkey, x)
            x = jnp.reshape(x, (args.num_minibatches * args.gradient_accumulation_steps, -1) + x.shape[1:])
            return x

        flatten_storage = jax.tree_map(flatten, storage)
        flatten_advantages = flatten(advantages)
        flatten_target_values = flatten(target_values)
        shuffled_storage = jax.tree_map(convert_data, flatten_storage)
        shuffled_advantages = convert_data(flatten_advantages)
        shuffled_target_values = convert_data(flatten_target_values)

        def update_minibatch(agent_state, minibatch):
            mb_obs, mb_actions, mb_behavior_logprobs, mb_advantages, mb_target_values = minibatch
            (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                agent_state.params,
                mb_obs,
                mb_actions,
                mb_behavior_logprobs,
                mb_advantages,
                mb_target_values,
            )
            grads = jax.lax.pmean(grads, axis_name="local_devices")
            agent_state = agent_state.apply_gradients(grads=grads)
            return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl)

        agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
            update_minibatch,
            agent_state,
            (
                shuffled_storage.obs,
                shuffled_storage.actions,
                shuffled_storage.logprobs,
                shuffled_advantages,
                shuffled_target_values,
            ),
        )
        return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl)

    (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
        update_epoch, (agent_state, key), (), length=args.update_epochs
    )
    loss = jax.lax.pmean(loss, axis_name="local_devices").mean()
    pg_loss = jax.lax.pmean(pg_loss, axis_name="local_devices").mean()
    v_loss = jax.lax.pmean(v_loss, axis_name="local_devices").mean()
    entropy_loss = jax.lax.pmean(entropy_loss, axis_name="local_devices").mean()
    approx_kl = jax.lax.pmean(approx_kl, axis_name="local_devices").mean()
    return agent_state, key, {
        "loss": loss, 
        "policy_loss": pg_loss, 
        "value_loss": v_loss,
        "entropy": entropy_loss, 
        "approx_kl": approx_kl
    }


@jax.jit
def compute_gae(
    agent_state: TrainState,
    next_obs: np.ndarray,
    next_done: np.ndarray,
    storage: structs.Transition,
):
    next_value = policy.get_value(agent_state.params, next_obs)

    advantages = jnp.zeros_like(next_value)
    dones = jnp.concatenate([jnp.logical_or(storage.terminations, storage.truncations), next_done[None, :]], axis=0)
    values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
    _, advantages = jax.lax.scan(
        _compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
    )
    return advantages, advantages + storage.values


def _compute_gae_once(carry, inp):
    advantages = carry
    nextdone, nextvalues, curvalues, reward = inp
    nextnonterminal = 1.0 - nextdone

    delta = reward + args.gamma * nextvalues * nextnonterminal - curvalues
    advantages = delta + args.gamma * args.gae_lambda * nextnonterminal * advantages
    return advantages, advantages


def ppo_loss(params, obs, actions, behavior_logprobs, advantages, target_values):
    newlogprob, entropy, newvalue = policy.get_logprob_entropy_value(params, obs, actions)
    logratio = newlogprob - behavior_logprobs
    ratio = jnp.exp(logratio)
    approx_kl = ((ratio - 1) - logratio).mean()

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss
    v_loss = 0.5 * ((newvalue - target_values) ** 2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))