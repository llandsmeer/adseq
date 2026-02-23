# ADSEQ: Autodifferentiable spike-event queues for brain simulation on AI accelerators

![ADSEQ aims to implement delay-enabled differentiable event queues for brain simulations](https://github.com/llandsmeer/ml_spike_event_queues/blob/main/img/adseq.png?raw=true)

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

