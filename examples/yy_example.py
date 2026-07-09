from functools import partial
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
import adseq.bridges.flax_bridge as adseq

dt = 0.05
T = 250
T_max = T * dt
weight_init = partial(jax.random.uniform, minval=-0.1, maxval=1.5)
syn = dict(tau_syn1_ms=0.25, tau_syn2_ms=1., max_delay=6., queue=adseq.implementations.SingleSpike)
tau = dict(tau_mem=5., reset_gradient='exact')

model = adseq.Sequential([
    adseq.DenseInput(dt, 120, weight_init=weight_init, **syn),
    adseq.LIF(dt, output='single_spike', **tau),
    adseq.Dense(dt, 3, weight_init=nn.initializers.uniform(1.5), **syn),
    adseq.LIF(dt, output='ttfs_and_spike', **tau),
])

def load_yy(path):
    with h5py.File(path, 'r', locking=False) as f:
        times, units, labels = f['spikes/times'][()], f['spikes/units'][()], f['labels'][()]
    xs = np.zeros((len(labels), T, 4), np.float32)
    xs[np.arange(len(labels))[:, None], times, units] = 1.
    return jnp.array(xs), jnp.array(labels)
xs_train, ys_train = load_yy('examples/yy_train.h5')
xs_test, ys_test = load_yy('examples/yy_test.h5')

params = model.init(jax.random.key(0), None, jnp.zeros(4))

def loss(params, x, y):
    ttfs, spike = (o[-1] for o in model.apply(params, x, method='trace'))
    ttfs = jnp.where(ttfs < 0, T_max, ttfs)
    l = optax.softmax_cross_entropy_with_integer_labels(-ttfs, y) + ((1 - spike) ** 2).sum()
    return l, jnp.argmin(ttfs)

optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(2e-3))
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, x, y):
    def total(p):
        l, pred = jax.vmap(loss, (None, 0, 0))(p, x, y)
        return l.mean(), (pred == y).mean()
    (l, acc), grad = jax.value_and_grad(total, has_aux=True)(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    return optax.apply_updates(params, updates), opt_state

@jax.jit
def accuracy(params, x, y):
    return (jax.vmap(loss, (None, 0, 0))(params, x, y)[1] == y).mean()

rng = np.random.default_rng(0)
nb = len(xs_train) // 64
for epoch in range(120):
    for idx in rng.permutation(len(xs_train))[:nb * 64].reshape(nb, 64):
        params, opt_state = step(params, opt_state, xs_train[idx], ys_train[idx])
    if epoch % 5 == 0:
        print(f'epoch {epoch:4d}  test_acc={float(accuracy(params, xs_test, ys_test)):.3f}')
