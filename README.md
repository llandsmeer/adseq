# ADSEQ: Autodifferentiable spike-event queues for brain simulation on AI accelerators

![ADSEQ aims to implement delay-enabled differentiable event queues for brain simulations](https://github.com/llandsmeer/ml_spike_event_queues/blob/main/img/adseq.png?raw=true)

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

## Citation

```bibtex
@article{landsmeer2025eventqueues,
  title={EventQueues: Autodifferentiable spike event queues for brain simulation on AI accelerators},
  author={Landsmeer, Lennart PL and Movahedin, Amirreza and Hamdioui, Said and Strydis, Christos},
  journal={arXiv preprint arXiv:2512.05906},
  year={2025}
}
```

