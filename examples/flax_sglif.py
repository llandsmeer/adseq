from functools import partial
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import adseq.bridges.flax_bridge as adseq

dt = 0.05
model = adseq.Sequential([
        # Hidden layer 1
        adseq.DenseInput(dt, 2,
            weight_init=partial(jax.random.uniform, minval=-0.1, maxval=1.5),
            queue=adseq.implementations.SingleSpike),
        adseq.LIF(dt, output='single_spike'),
        # Hidden layer 2
        adseq.Dense(dt, 5,
            weight_init=partial(jax.random.uniform, minval=-0.1, maxval=1.5),
            queue=adseq.implementations.SingleSpike),
        adseq.LIF(dt, output='single_spike'),
        # Output layer
            adseq.Dense(dt, 2, weight_init=nn.initializers.uniform(1.5)),
        adseq.LIF(dt, output='ttfs_and_spike'),
        ])


# temporal xor over 200 timesteps
def generate(a, b):
    X = jnp.zeros((1000, 2))
    X = X.at[20*a, 0].set(1)
    X = X.at[20*b, 1].set(1)
    return X
xs = jnp.array([
       generate(0, 0),
       generate(1, 0),
       generate(0, 1),
       generate(1, 1) ])
ys = jnp.array([ 0, 1, 1, 0 ])


params = model.init(jax.random.key(0), None, jnp.zeros(xs[0].shape[1]))

def loss(params, x, y, out=False):
    trace = model.apply(params, x, method='trace')
    ttfs: jax.Array = trace[0][-1]
    S: jax.Array = trace[1][-1]
    loss = optax.softmax_cross_entropy_with_integer_labels(ttfs, y) + (1-S)**2
    jax.debug.print('{} {}', ttfs, loss)
    return loss, jnp.where(ttfs[0] == ttfs[1], 0.5, ttfs[0] < ttfs[1])


optimizer = optax.adam(learning_rate=1e-1)
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, x, y):
    def batched_loss(params):
        ls, o = jax.vmap(loss, in_axes=[None, 0, 0])(params, x, y)
        return ls.sum(), (o == y).mean()
    (l, o), (g, _o) = batched_loss(params), \
                      jax.jacrev(batched_loss, has_aux=True)(params)
    updates, opt_state = optimizer.update(g, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, l, g, o

o = model.apply(params, xs[0], output_all=True, method='trace')
#isyn0, p

for epoch in range(1000):
    print()
    correct = 0
    half = 0
    trace = []
    params, opt_state, l, g, o = step(params, opt_state, xs, ys)
    #correct += ys[i] == o
    #half += 0.5 == o
    #trace.append(o)
        #print(g)
    #print(100 * correct / 4, '% |', *trace, '|', ys, '(', half, ')')
    # print(g)
    print(o, l)

