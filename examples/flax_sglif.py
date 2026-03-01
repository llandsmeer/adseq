import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from adseq.bridges import flax_bridge

dt = 0.25
model = flax_bridge.Sequential([
        flax_bridge.DenseInput(dt, 4),
        flax_bridge.SurrogateLIF(dt),
        flax_bridge.Dense(dt, 4),
        flax_bridge.SurrogateLIF(dt),
        flax_bridge.Dense(dt, 1),
        ])


params = model.init(jax.random.key(0), None, jnp.zeros(2))

def generate(a, b):
    X = jnp.zeros((1000, 3))
    X = X.at[0, 0].set(a)
    X = X.at[0, 1].set(b)
    X = X.at[0, 2].set(1)
    return X

xs = [
        generate(0, 0),
        generate(1, 0),
        generate(0, 1),
        generate(1, 1),
    ]

x = xs[0]
Y = model.apply(params, x, method='trace')

