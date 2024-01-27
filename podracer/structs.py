from __future__ import annotations

from typing import List, NamedTuple, Sequence
import jax
import jax.numpy as jnp


class Transition(NamedTuple):
    obs: list
    actions: list
    logprobs: list
    values: list
    rewards: list
    truncations: list
    terminations: list


    @staticmethod
    def _prepare_data(storage: List[Transition], devices: Sequence[jax.Device]) -> Transition:
        return jax.tree_map(
            lambda *xs: jnp.split(jnp.stack(xs), len(devices), axis=1),
            *storage
        )

    @staticmethod
    def make_sharded(storage: List[Transition], devices: Sequence[jax.Device]) -> Transition:
        storage = Transition._prepare_data(storage, devices)
        return Transition(*list(map(
            lambda x: jax.device_put_sharded(x, devices=devices),
            storage
        )))