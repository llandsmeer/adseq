#!/home/llandsmeer/repos/llandsmeer/ml_spike_event_queues/benchmarks/env/bin/python

import matplotlib.pyplot as plt

import optax
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jaxley as jx
import adseq.bridges.jaxley_bridge as adseq

dt = 0.025

## Set up a network with two multicompartmental HH neurons
cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=4), parents=[-1, 0, 0, 1, 1, 2, 2])
net = jx.Network([cell, cell])
net.insert(jx.channels.Na())
net.insert(jx.channels.K())
net.insert(jx.channels.Leak())
net.cell(0).branch(0).loc(0.0).stimulate(
    jx.step_current(i_delay=8.0, i_dur=2.0, i_amp=0.05, delta_t=dt, t_max=50.))
net.cell([0,1]).branch(0).loc(0.0).record()

## Insert delayed synapses
jx.connect(net.cell(0).branch(0).loc(0.0),
           net.cell(1).branch(0).loc(0.0),
           adseq.DelaySynapse(vthres=10.0))
net.set('DelaySynapse_delay', 10.0)
net.set('DelaySynapse_weight', 0.05)
net.make_trainable('DelaySynapse_delay')


## Set up parameters
parameters = net.get_parameters()
transform = jx.ParamTransform([
    {'DelaySynapse_delay':  jx.optimize.transforms.SigmoidTransform(.1, 50.0)},
])

# Plot gradient
# def trace(delay):
#     return jx.integrate(net, delta_t=dt, t_max=100, params=[{'DelaySynapse_delay': jnp.array([delay])}])[1]
# g1 = jax.jacfwd(trace)(10.0)
# g2 = jax.jacrev(trace)(10.0)

# Set up training
def loss(opt_params):
    s = jx.integrate(net, delta_t=dt, params=transform.forward(opt_params))
    t = jnp.linspace(0, 1, s.shape[1])
    conv = jnp.exp(-10*(t - 0.7)**2)
    return - (s[1] * conv).mean()

opt_params = transform.inverse(parameters)
optimizer = optax.adam(learning_rate=1e-1)
opt_state = optimizer.init(opt_params)
#g = jax.jit(jax.value_and_grad(loss, argnums=0))

@jax.jit
def step(opt_params, opt_state):
    #loss, gradient = g(opt_params)
    l, gradient = loss(opt_params), jax.jacfwd(loss)(opt_params)
    updates, opt_state = optimizer.update(gradient, opt_state)
    opt_params = optax.apply_updates(opt_params, updates)
    return l, opt_params, opt_state, gradient

## Optimize
s_old = jx.integrate(net, delta_t=dt, params=transform.forward(opt_params))
for i in range(100):
    loss, opt_params, opt_state, g = step(opt_params, opt_state)
    print(i, loss, g, opt_params)
s_new = jx.integrate(net, delta_t=dt, params=transform.forward(opt_params))

## Plot
t = jnp.arange(s_old.shape[1]) * dt
plt.plot(t, s_old.T, color='black')
plt.plot(t, s_new.T-100, color='black')
plt.show()

