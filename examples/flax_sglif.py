from functools import partial
import jax, jax.numpy as jnp, optax
import flax.linen as nn
import adseq.bridges.flax_bridge as adseq

dt = 0.05
T = 400
T_max = T * dt
winit = partial(jax.random.uniform, minval=-1., maxval=1.5)

model = adseq.Sequential([
    adseq.DenseInput(dt, 20, weight_init=winit, queue=adseq.implementations.SingleSpike, max_delay=6.),
    adseq.LIF(dt, output='single_spike', reset_gradient='exact'),
    adseq.Dense(dt, 20, weight_init=winit, queue=adseq.implementations.SingleSpike, max_delay=6.),
    adseq.LIF(dt, output='single_spike', reset_gradient='exact'),
    adseq.Dense(dt, 2, weight_init=winit, max_delay=6.),
    adseq.LIF(dt, output='ttfs_and_spike', reset_gradient='exact'),
])

# temporal XOR
def sample(a, b):
    X = jnp.zeros((T, 2))
    X = X.at[20 if a else 120, 0].set(1.)
    X = X.at[20 if b else 120, 1].set(1.)
    return X
xs = jnp.stack([sample(0, 0), sample(0, 1), sample(1, 0), sample(1, 1)])
ys = jnp.array([0, 1, 1, 0])

params = model.init(jax.random.key(0), None, jnp.zeros(2))

def loss(params, x, y):
    ttfs, spike = (o[-1] for o in model.apply(params, x, method='trace'))
    ttfs = jnp.where(ttfs < 0, T_max, ttfs)                        # -1 means "never fired"
    # correct output neuron should fire >=2ms before the other; keep neurons spiking
    l = jnp.maximum(0., ttfs[y] - ttfs[1 - y] + 2.) + ((1 - spike) ** 2).sum()
    return l, jnp.argmin(ttfs)

optimizer = optax.adam(1e-2)
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state):
    def total(p):
        l, pred = jax.vmap(loss, (None, 0, 0))(p, xs, ys)
        return l.sum(), (pred == ys).mean()
    (l, acc), grad = jax.value_and_grad(total, has_aux=True)(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    return optax.apply_updates(params, updates), opt_state, l, acc

for epoch in range(200):
    params, opt_state, l, acc = step(params, opt_state)
    if epoch % 20 == 0:
        print(f'epoch {epoch:4d}  loss={float(l):.4f}  acc={float(acc):.2f}')
