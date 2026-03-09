# ADSEQ: A library for Autodifferentiable spike-event queues for brain simulation on AI accelerators

![ADSEQ aims to implement delay-enabled differentiable event queues for brain simulations](https://github.com/llandsmeer/ml_spike_event_queues/blob/main/img/adseq.png?raw=true)

## Installation

```
pip install adseq
```

## RSNN example

```python
Q = adseq.implementations.FIFORing.sized(3)
syn = adseq.synapse.mk_synapses(Q,
      delay_ms=delays, dt_ms=dt,
      vthres=vthres, tau_syn_ms=tau_syn, n=n*n,
      max_delay_ms=7
      )
syn_step = jax.jit(type(syn).timestep_spike_detect_pre)
v = jnp.zeros(n)
state = v, syn
def step(state, t):
    v, syn = state
    isyn = weight @ syn.isyn.reshape((n,n)).sum(0)
    vnext, s = lif_step(v, isyn, tau_mem, dt, vthres)
    syn = syn.timestep_spike_detect_pre(
                   ts=t,
                   v=jnp.repeat(v, n),
                   vnext=jnp.repeat(vnext, n))
    state = vnext, syn
    return state, (v, s)
_, trace = jax.lax.scan(step, state, xs=ts = jnp.arange(10000)*dt)
```
# Bridges

`adseq` does not aim to be another brain simulation package. Instead, it provides 'bridges' to other existing packages.

## Jaxley example

Support is a bit experimental.
Note that performance will be suboptimal, and limited to singlespike implemetations, until [Jaxley PR646](https://github.com/jaxleyverse/jaxley/issues/632) is merged

```python
import jaxley as jx
import adseq.bridges.jaxley_bridge as adseq

# [...]

net = jx.Network([cell, cell])
jx.connect(net.cell(0).branch(0).loc(0.0),
           net.cell(1).branch(0).loc(0.0),
           adseq.DelaySynapse(vthres=10.0))
net.set('DelaySynapse_delay', 10.0)
net.set('DelaySynapse_weight', 0.05)
net.make_trainable('DelaySynapse_delay')

# [...]

parameters = net.get_parameters()
transform = jx.ParamTransform([
    {'DelaySynapse_delay':  jx.optimize.transforms.SigmoidTransform(.1, 50.0)},
])
def loss(opt_params):
    s = jx.integrate(net, delta_t=dt, params=transform.forward(opt_params))
    t = jnp.linspace(0, 1, s.shape[1])
    conv = jnp.exp(-10*(t - 0.7)**2) # move spike towards 70% of simulation time
    return - (s[1] * conv).mean()
for i in range(100):
    l, gradient = loss(opt_params), jax.jacfwd(loss)(opt_params)
    updates, opt_state = optimizer.update(gradient, opt_state)
    opt_params = optax.apply_updates(opt_params, updates)
```

## Flax bridge

```python
import flax.linen as nn
import adseq.bridges.flax_bridge as adseq

dt = 0.1
model = adseq.Sequential([
        # single spike hidden layer
        adseq.DenseInput(dt, 30, queue=adseq.implementations.SingleSpike),
        adseq.LIF(dt, output='single_spike'),

        # ttfs output layer
        adseq.Dense(dt, 2, weight_init=nn.initializers.uniform(1.5)),
        adseq.LIF(dt, output='ttfs'),
        ])

x = jnp.zeros((100, 4))
x = x.at[0, 0].set(1)
x = x.at[100, 1].set(1)
x = x.at[50, 2].set(1)
x = x.at[10, 3].set(1)

params = model.init(jax.random.key(0), None, x)
ttfs = model.apply(params, x, method='trace')
```

# Module Documentation

 - [docs](https://llandsmeer.github.io/adseq/index.html)
     - [synapse2](https://llandsmeer.github.io/adseq/synapse2.html)
     - [synapse](https://llandsmeer.github.io/adseq/synapse.html)
     - [bridges](https://llandsmeer.github.io/adseq/bridges/index.html)
         - [jaxley_bridge](https://llandsmeer.github.io/adseq/bridges/jaxley_bridge.html)
         - [flax_bridge](https://llandsmeer.github.io/adseq/bridges/flax_bridge.html)
     - [benchmarks](https://llandsmeer.github.io/adseq/benchmarks/index.html)
         - [profile_poisson](https://llandsmeer.github.io/adseq/benchmarks/profile_poisson.html)
         - [profile_recurrent_snn](https://llandsmeer.github.io/adseq/benchmarks/profile_recurrent_snn.html)
         - [profile_jaxley](https://llandsmeer.github.io/adseq/benchmarks/profile_jaxley.html)
         - [loss2](https://llandsmeer.github.io/adseq/benchmarks/loss2.html)
         - [profile_lossy_queues](https://llandsmeer.github.io/adseq/benchmarks/profile_lossy_queues.html)
     - [implementations](https://llandsmeer.github.io/adseq/implementations/index.html)
         - [bgpq1](https://llandsmeer.github.io/adseq/implementations/bgpq1.html)
         - [fifo_ring](https://llandsmeer.github.io/adseq/implementations/fifo_ring.html)
         - [ring](https://llandsmeer.github.io/adseq/implementations/ring.html)
         - [single_spike](https://llandsmeer.github.io/adseq/implementations/single_spike.html)
         - [sorted_array](https://llandsmeer.github.io/adseq/implementations/sorted_array.html)
         - [single_spike_keep](https://llandsmeer.github.io/adseq/implementations/single_spike_keep.html)
         - [lossy_ring](https://llandsmeer.github.io/adseq/implementations/lossy_ring.html)
         - [bitarray32](https://llandsmeer.github.io/adseq/implementations/bitarray32.html)
         - [binary_heap](https://llandsmeer.github.io/adseq/implementations/binary_heap.html)
         - [do_nothing](https://llandsmeer.github.io/adseq/implementations/do_nothing.html)
         - [base](https://llandsmeer.github.io/adseq/implementations/base.html)

## Citation

```bibtex
@article{landsmeer2025eventqueues,
  title={EventQueues: Autodifferentiable spike event queues for brain simulation on AI accelerators},
  author={Landsmeer, Lennart PL and Movahedin, Amirreza and Hamdioui, Said and Strydis, Christos},
  journal={arXiv preprint arXiv:2512.05906},
  year={2025}
}
```

