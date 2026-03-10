# ADSEQ: A library for Autodifferentiable spike-event queues for brain simulation on AI accelerators

![ADSEQ aims to implement delay-enabled differentiable event queues for brain simulations](https://github.com/llandsmeer/ml_spike_event_queues/blob/main/img/adseq.png?raw=true)

## Installation

```
pip install adseq
```

## Usage

ADSEQ consists of three layers: queue implementations, synapses and bridges.

The queue implementations are simple queues with an enqueue() and pop() function, that are carefully written such that tangents are retained alongside primal values in the queue; i.e. they are autodifferentiable.

Synapsesbuild on top of these to provide single or double exponential synapses that correctly handle differentiation though a voltage-threshold based spike detector.

Finally bridges allow using these primitives within other frameworks.

### Queues

All queues implement the BaseQueue protocol

```
class BaseQueue(Protocol):
    @classmethod
    def init(cls, delay: int|None) -> Self: ...
    def enqueue(self, n: int) -> Self: ...
    def pop(self, n: int) -> Tuple[Self, int | jax.Array]: ...
```

For the configurable capacity limited queues (`FIFORing`, `SortedArray`, `LossyRing` and `BinaryHeap`), a `.sized(n)` classmethod exists to create the specified capacity-limited queue. The `delay` argument to init() specifies the maximum queue capacity in the case of `Ring`.


| Implementation  | Usecase/Platform | Limitations |
| --------------- | -------------------- | ----------------- |
| DoNothing       | Testing              | no spikes at all                |
| SingleSpike     | TTFS                 | only single spike                |
| SingleSpikeKeep | Performance          | only single spike               |
| SortedArray     | TPU                  |                |
| BGPQ1           | --                   |                |
| LossyRing       | --                   |                |
| BinaryHeap      | CPU                  |                |
| FIFORing        | CPU/GPU              | only homogeneous delays               |
| Ring            | GPU                  | maximum delay               |
| BitArray32      | *                    | fixed 32 spike capacity, no gradients                |


### Synapses


Synapses (`mk_synapse()` and `mk_synapse2()`) provide one function `timestep_spike_detect_pre()` which is supposed to be called every timestep, and a synaptic current output property `isyn`.


If you have more than one neuron (the usual case), the constructors `mk_synapses()` and `mk_synapse2s()` provide

The detection function relies on `v` and `vnext`, for both the detection of a spike and estimating dv/dt for gradient calculation.

#### Single synapse

Example:
```
syn = adseq.mk_synapse2(
        adseq.SingleSpike,
        dt_ms=1.0,
        vthres=1.0,
        tau_syn1_ms=1.0,
        tau_syn2_ms=10.0,
        max_delay_ms=100.,
        )
a = 1.0
d = 1.0
syn = syn.timestep_spike_detect_pre(t_ms=0, v=1.0-a*0.1, vnext=1.0+a*0.1, delay_ms=d)
print(syn.isyn)
# 0.0
syn = syn.timestep_spike_detect_pre(t_ms=1, v=0.9, vnext=0.9, delay_ms=0.0)
print(syn.isyn)
# 0.0
syn = syn.timestep_spike_detect_pre(t_ms=2, v=0.9, vnext=0.9, delay_ms=0.0)
print(syn.isyn)
# 0.7705642
```

In this example, the synaptic current should be autodifferentiable toward the voltage slope `a` and delay `d`.

#### RSNN example

Synapses can be used to construct networks.

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

### Bridges

`adseq` does not aim to be another brain simulation package. Instead, it provides 'bridges' to other existing packages.

#### Jaxley example

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

#### Flax bridge

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

DOCS

## Citation

```bibtex
@article{landsmeer2025eventqueues,
  title={EventQueues: Autodifferentiable spike event queues for brain simulation on AI accelerators},
  author={Landsmeer, Lennart PL and Movahedin, Amirreza and Hamdioui, Said and Strydis, Christos},
  journal={arXiv preprint arXiv:2512.05906},
  year={2025}
}
```

