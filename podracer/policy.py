from typing import Sequence
import flax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np
from podracer.args import args


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        return x + inputs
    

class ConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x


class SharedNetwork(nn.Module):
    channels: Sequence[int]
    hiddens: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for channels in self.channels:
            x = ConvSequence(channels)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        for hidden in self.hiddens:
            x = nn.Dense(hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
        return x


class CriticHead(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x).squeeze(-1)


class ActorHead(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
    

class PolicyNetwork(nn.Module):
    action_dim: int
    channels: Sequence[int]
    hiddens: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = SharedNetwork(self.channels, self.hiddens)(x)
        logits = ActorHead(self.action_dim)(x)
        value = CriticHead()(x)
        return logits, value


@jax.jit
def get_action_and_value(
    params: flax.core.FrozenDict,
    obs: np.ndarray,
    key: jax.random.PRNGKey,
):
    obs = jnp.array(obs)
    logits, value = PolicyNetwork(args.action_size, args.channels, args.hiddens).apply(params, obs)
    # sample action: Gumbel-softmax trick
    # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=logits.shape)
    action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
    logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
    return action, logprob, value, key

@jax.jit
def get_value(
    params: flax.core.FrozenDict,
    obs: np.ndarray,
):
    _, value = PolicyNetwork(args.action_size, args.channels, args.hiddens).apply(params, obs)
    return value

@jax.jit
def get_logprob_entropy_value(
    params: flax.core.FrozenDict,
    obs: np.ndarray,
    actions: np.ndarray,
):
    logits, value = PolicyNetwork(args.action_size, args.channels, args.hiddens).apply(params, obs)
    logprob = jax.nn.log_softmax(logits)[jnp.arange(actions.shape[0]), actions]
    logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    logits = logits.clip(min=jnp.finfo(logits.dtype).min)
    p_log_p = logits * jax.nn.softmax(logits)
    entropy = -p_log_p.sum(-1)
    return logprob, entropy, value
