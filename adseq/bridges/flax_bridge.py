import flax.linen as nn
import abc

import typing
import jax

import jax.numpy as jnp

from .. import implementations
from .. import synapse2

if not hasattr(typing, 'Self'): typing.Self = None # type: ignore

class Sequential(nn.Module):
    layers: typing.List[nn.Module]

    def init_carry(self, x):
        cs = []
        for layer in self.layers:
            if hasattr(layer, 'init_carry'):
                carry = layer.init_carry(x)
                _carry, x = layer(carry, x)
            else:
                carry = None
                x = layer(x)
            cs.append(carry)
        return cs

    def trace(self, xs):
        carry = self.init_carry(xs[0])
        carry, ys = jax.lax.scan(self, carry, xs)
        return ys

    def __call__(self, carry, x):
        if carry is None: carry = self.init_carry(x)
        carry_out = []
        for c, layer in zip(carry, self.layers):
            if c is None:
                x = layer(x)
            else:
                c, x = layer(c, x)
            carry_out.append(c)
        return carry_out, x


class StaticMultiSynapse(abc.ABC):
    @property
    @abc.abstractmethod
    def isyn(self) -> jax.Array: ...

    @abc.abstractmethod
    def timestep_spike_detect_pre(self, ts, v, vnext, delay_ms) -> typing.Self: ... # type: ignore

    @abc.abstractmethod
    def timestep_static_spike(self, ts, s, delay_ms) -> typing.Self: ... # type: ignore

DelaySynapseCarry: typing.TypeAlias = tuple[StaticMultiSynapse, int] | tuple[StaticMultiSynapse, int, jax.Array]


class DenseInput(nn.Module):
    dt: float
    nout: int | None = None
    def setup(self):
        self.model = Sequential([
            Explode(self.nout),
            DelayedStaticSynapse(self.dt),
            LTIReduce(self.nout)
            ])
    def init_carry(self, x):
        return self.model.init_carry(x)

    def __call__(self, carry, x):
        return self.model(carry, x)

class Dense(nn.Module):
    dt: float
    nout: int | None = None
    def setup(self):
        self.model = Sequential([
            Explode(self.nout),
            DelayedThresholdSynapse(self.dt),
            LTIReduce(self.nout)
            ])
    def init_carry(self, x):
        return self.model.init_carry(x)

    def __call__(self, carry, x):
        return self.model(carry, x)


class Explode(nn.Module):
    'HelperModule for Dense LTI synapses; duplicate across nout targets'
    nout: int | None = None
    @nn.compact
    def __call__(self, v: jax.Array):
        assert len(v.shape) == 1
        nout = self.nout if self.nout is not None else v.shape[-1]
        return jnp.tile(v, nout)

class LTIReduce(nn.Module):
    'HelperModule for Dense LTI synapses; weighted sum up to nout features'
    nout: int | None = None
    weight_init: nn.initializers.Initializer = nn.initializers.normal()
    @nn.compact
    def __call__(self, isyn: jax.Array):
        nsyn = isyn.shape[-1]
        if self.nout is not None:
            nin = nsyn // self.nout
            nout = self.nout
        else:
            nin = nout = int(nsyn ** 0.5)
        assert nsyn == nin * nout
        assert len(isyn.shape) == 1
        weight = self.param('weight', self.weight_init, isyn.shape)
        isyn = (isyn * weight).reshape(nout, nin)
        return isyn.sum(1) # second dimension if not batched

class DelayedThresholdSynapse(nn.Module):
    'Plain delayed synapse that detects spikes given input voltage'
    dt: float
    tau_syn1_ms: float = 0.5
    tau_syn2_ms: float = 2.0
    max_delay: float = 200.
    vthres: float = 1.0
    delay_init: nn.initializers.Initializer = nn.initializers.normal()
    delay_activation = lambda self, x: self.max_delay * (1 + nn.tanh(x))
    queue: type[implementations.BaseQueue] = implementations.FIFORing.sized(4) # type: ignore

    def init_carry(self, v=jax.Array, vnext: jax.Array|None=None) -> DelaySynapseCarry:
        'if vnext is none, we delay one timestep'
        assert len(v.shape) == 1
        syn = synapse2.mk_synapse2s(self.queue,
                                   vthres=self.vthres,
                                   tau_syn1_ms=self.tau_syn1_ms,
                                   tau_syn2_ms=self.tau_syn2_ms,
                                   dt_ms=self.dt,
                                   n=len(v),
                max_delay_ms=self.max_delay)
        if vnext is None:
            return syn, 0, 0*v
        else:
            assert vnext.shape == v.shape
            return syn, 0 # type: ignore

    @nn.compact
    def __call__(self, carry: DelaySynapseCarry, v: jax.Array, vnext: jax.Array|None=None) -> tuple[DelaySynapseCarry, jax.Array]:
        if vnext is None:
            assert len(carry) == 3
            syn, ts, vprev = carry
            v, vnext = vprev, v
        else:
            assert len(carry) == 2
            syn, ts = carry
        delay = self.delay_activation(self.param('delay', self.delay_init, v.shape))
        isyn = syn.isyn
        syn = syn.timestep_spike_detect_pre(ts=ts, v=v, vnext=vnext, delay_ms=delay)
        if len(carry) == 3:
            return (syn, ts+1, vnext), isyn
        else:
            return (syn, ts+1), isyn

class DelayedStaticSynapse(nn.Module):
    'Plain delayed synapse for static data (non neuron produced) delays'
    dt: float
    tau_syn1_ms: float = 0.5
    tau_syn2_ms: float = 2.0
    max_delay: float = 200.
    delay_init: nn.initializers.Initializer = nn.initializers.normal()
    delay_activation = lambda self, x: self.max_delay * (1 + nn.tanh(x))
    queue: type[implementations.BaseQueue] = implementations.FIFORing.sized(4) # type: ignore

    def init_carry(self, s) -> DelaySynapseCarry:
        'Example input s'
        syn = synapse2.mk_synapse2s(self.queue,
                                   vthres=0.0,
                                   tau_syn1_ms=self.tau_syn1_ms,
                                   tau_syn2_ms=self.tau_syn2_ms,
                                   dt_ms=self.dt,
                                   n=len(s),
                max_delay_ms=self.max_delay)
        return syn, 0

    @nn.compact
    def __call__(self, carry: DelaySynapseCarry, s: jax.Array) -> tuple[DelaySynapseCarry, jax.Array]:
        's is a binary indicator for spikes (1 meaning spike, 0 no input spike)'
        assert len(carry) == 2
        syn, ts = carry
        delay = self.delay_activation(self.param('delay', self.delay_init, s.shape))
        isyn = syn.isyn
        syn = syn.timestep_static_spike(ts=ts, s=s, delay_ms=delay)
        return (syn, ts+1), isyn

LIFCarry: typing.TypeAlias = jax.Array

class SurrogateLIF(nn.Module):
    dt: float
    tau_mem: float = 10.
    vthres: float = 1.0

    def init_carry(self, isyn):
        return isyn*0

    @nn.compact
    def __call__(self, carry: LIFCarry, isyn: jax.Array) -> tuple[LIFCarry, jax.Array]:
        v = carry
        S = superspike(v - self.vthres)
        beta = jnp.exp(-self.dt/self.tau_mem)
        v_next = (1 - S) * (beta * v + isyn*self.dt)
        return v_next, S


@jax.custom_jvp
def superspike(x):
    'doi.dx/10.1162/neco_a_01086'
    return jnp.where(x < 0, 0.0, 1.0)

@superspike.defjvp
def superspike_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    primal_out = jnp.where(x < 0, 0.0, 1.0)
    tangent_out = x_dot / (jnp.abs(x)+1)**2
    return primal_out, tangent_out

