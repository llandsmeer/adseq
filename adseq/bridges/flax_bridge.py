import flax.linen as nn
import abc

import typing
import jax

import jax.numpy as jnp

from .. import implementations
from .. import synapse2

if not hasattr(typing, 'Self'): typing.Self = None # type: ignore

class StaticMultiSynapse(abc.ABC):
    @property
    @abc.abstractmethod
    def isyn(self) -> jax.Array: ...
    @abc.abstractmethod
    def timestep_spike_detect_pre(self, t_ms, v, vnext, delay_ms) -> typing.Self: ... # type: ignore
    @abc.abstractmethod
    def timestep_static_spike(self, t_ms, s, delay_ms) -> typing.Self: ... # type: ignore


TTFSCarry: typing.TypeAlias = tuple[jax.Array, int] | tuple[jax.Array, int, jax.Array]
LIFCarry: typing.TypeAlias = jax.Array
DelaySynapseCarry: typing.TypeAlias = tuple[StaticMultiSynapse, int] | tuple[StaticMultiSynapse, int, jax.Array]


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

    def trace(self, xs, output_all=False):
        carry = self.init_carry(xs[0])
        carry, ys = jax.lax.scan(lambda c, x: self.__call__(c, x, output_all), carry, xs)
        return ys

    def __call__(self, carry, x, output_all=False):
        if carry is None: carry = self.init_carry(x)
        carry_out = []
        if output_all:
            output = []
        for c, layer in zip(carry, self.layers):
            if c is None:
                x = layer(x)
            else:
                c, x = layer(c, x)
            carry_out.append(c)
            if output_all:
                output.append(x)
        if output_all:
            return carry_out, output
        return carry_out, x






class DenseInput(nn.Module):
    dt: float
    nout: int | None = None
    weight_init: nn.initializers.Initializer= nn.initializers.uniform(1.5)
    delay_init: nn.initializers.Initializer = nn.initializers.normal(1.)
    queue: type[implementations.BaseQueue] = implementations.FIFORing.sized(4) # type: ignore
    max_delay: float = 20.
    def setup(self):
        self.model = Sequential([
            Explode(self.nout),
            DelayedStaticSynapse(self.dt, delay_init=self.delay_init, queue=self.queue, max_delay=self.max_delay),
            LTIReduce(self.nout, self.weight_init)
            ])
    def init_carry(self, x):
        return self.model.init_carry(x)

    def __call__(self, carry, x):
        return self.model(carry, x)

class Dense(nn.Module):
    dt: float
    nout: int | None = None
    weight_init: nn.initializers.Initializer= nn.initializers.uniform(1.5)
    delay_init: nn.initializers.Initializer = nn.initializers.normal(1.)
    queue: type[implementations.BaseQueue] = implementations.FIFORing.sized(4) # type: ignore
    max_delay: float = 20.
    def setup(self):
        self.model = Sequential([
            Explode(self.nout),
            DelayedThresholdSynapse(self.dt, delay_init=self.delay_init, queue=self.queue, max_delay=self.max_delay),
            LTIReduce(self.nout, self.weight_init)
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
        # isyn = (isyn * weight).reshape(nout, nin)
        isyn = (isyn * weight).reshape(nout, nin)
        return isyn.sum(1) # second dimension if not batched

class DelayedThresholdSynapse(nn.Module):
    'Plain delayed synapse that detects spikes given input voltage'
    dt: float
    tau_syn1_ms: float = 0.5
    tau_syn2_ms: float = 2.0
    max_delay: float = 20.
    vthres: float = 1.0
    delay_init: nn.initializers.Initializer = nn.initializers.normal(1.)
    delay_activation = lambda self, x: self.max_delay * (1 + nn.tanh(x))
    queue: type[implementations.BaseQueue] = implementations.FIFORing.sized(4) # type: ignore

    def init_carry(self, v:jax.Array, vnext: jax.Array|None=None) -> DelaySynapseCarry:
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
            return syn, 0, 0*v # type: ignore
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
        syn = syn.timestep_spike_detect_pre(t_ms=self.dt*ts, v=v, vnext=vnext, delay_ms=delay)
        if len(carry) == 3:
            return (syn, ts+1, vnext), isyn
        else:
            return (syn, ts+1), isyn

class DelayedStaticSynapse(nn.Module):
    'Plain delayed synapse for static data (non neuron produced) delays'
    dt: float
    tau_syn1_ms: float = 0.5
    tau_syn2_ms: float = 2.0
    max_delay: float = 20.
    delay_init: nn.initializers.Initializer = nn.initializers.normal(1.)
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
        syn = syn.timestep_static_spike(t_ms=self.dt*ts, s=s, delay_ms=delay)
        return (syn, ts+1), isyn

class LIF(nn.Module):
    dt: float
    tau_mem: float = 10.
    vthres: float = 1.0
    reset_gradient: typing.Literal['surrogate'] = 'surrogate'
    output: typing.Literal['voltage'] | typing.Literal['single_spike'] | typing.Literal['ttfs'] | typing.Literal['superspike'] | typing.Literal['ttfs_and_spike'] = 'voltage'

    def setup(self):
        assert self.reset_gradient == 'surrogate'
        self.model = SurrogateLIF(self.dt, self.tau_mem, self.vthres)
        if self.output == 'voltage':
            self.model_output = None
        elif self.output == 'superspike':
            self.model_output = SurrogateSpikeFilter(self.dt, self.vthres)
        elif self.output == 'single_spike':
            self.model_output = SingleSpikeFilter(self.dt, self.vthres)
        elif self.output == 'ttfs':
            self.model_output = TTFSFilter(self.dt, self.vthres)
        elif self.output == 'ttfs_and_spike':
            self.model_output = TTFSAndSpikeFilter(self.dt, self.vthres)

    def init_carry(self, isyn):
        carry = self.model.init_carry(isyn)
        if self.model_output is None:
            return carry
        _carry, v = self.model(carry, isyn)
        carry_output = self.model_output.init_carry(v)
        return carry, carry_output

    def __call__(self, carry, isyn):
        if self.model_output is None:
            carry, out = self.model(carry, isyn)
        else:
            c0, c1 = carry
            c0, v = self.model(c0, isyn)
            c1, out = self.model_output(c1, v)
            carry = c0, c1
        return carry, out


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
        return v_next, v

class SurrogateSpikeFilter(nn.Module):
    dt: float = None
    vthres: float = 1.0
    def init_carry(self, v): return None
    def __call__(self, carry, v):
        S = superspike(v - self.vthres)
        return carry, S

class SingleSpikeFilter(nn.Module):
    'Passthrough voltage until spike, then hold'
    dt: float = None
    vthres: float = 1.0

    def init_carry(self, v:jax.Array, vnext: jax.Array|None=None) -> TTFSCarry:
        'if vnext is none, we delay one timestep'
        assert len(v.shape) == 1
        if vnext is None:
            return 0*v, 0, 0*v
        else:
            assert vnext.shape == v.shape
            return 0*v, 0

    @nn.compact
    def __call__(self, carry: LIFCarry, v: jax.Array, vnext: jax.Array|None=None) -> tuple[TTFSCarry, jax.Array]:
        if vnext is None:
            assert len(carry) == 3
            vhold, ts, vprev = carry
            v, vnext = vprev, v
        else:
            assert len(carry) == 2
            vhold, ts = carry
        out = jnp.where(vhold >= self.vthres, vhold, v)
        vhold = jnp.where(vhold >= self.vthres, vhold, v) # would prefer stop gradient here
        if len(carry) == 3:
            return (vhold, ts+1, vnext), out
        else:
            return (vhold, ts+1), out

class TTFSFilter(nn.Module):
    'Receives voltages, outputs differentiable first spike time'
    dt: float
    vthres: float = 1.0

    def init_carry(self, v:jax.Array, vnext: jax.Array|None=None) -> TTFSCarry:
        'if vnext is none, we delay one timestep'
        assert len(v.shape) == 1
        if vnext is None:
            return -1 + 0*v, 0, 0*v
        else:
            assert vnext.shape == v.shape
            return -1 + 0*v, 0

    @nn.compact
    def __call__(self, carry: TTFSCarry, v: jax.Array, vnext: jax.Array|None=None) -> tuple[TTFSCarry, jax.Array]:
        if vnext is None:
            assert len(carry) == 3
            ttfs, ts, vprev = carry
            v, vnext = vprev, v
        else:
            assert len(carry) == 2
            ttfs, ts = carry
        ttfs: jax.Array
        tpost = jax.vmap(synapse2.spike_detect, in_axes=[None,None,None,0,0,None])(self.dt, self.dt*ts, self.vthres, v, vnext, 0.)
        ttfs = jnp.where((ttfs != -1),
                   jnp.where((tpost != -1),
                             jnp.minimum(ttfs, tpost),
                             ttfs),
                         tpost)
        if len(carry) == 3:
            return (ttfs, ts+1, vnext), ttfs
        else:
            return (ttfs, ts+1), ttfs

class TTFSAndSpikeFilter(TTFSFilter):
    '''
    Receives voltages, outputs differentiable first spike time and surrogate-differentiable spike-indicator

    The main use case is that when there is no spike generated under TTFS, the spike-indicator can help the model generate a spike.
    '''

    def __call__(self, carry: TTFSCarry, v: jax.Array, vnext: jax.Array|None=None) -> tuple[TTFSCarry, tuple[jax.Array, jax.Array]]:
        carry, ttfs = super().__call__(carry, v, vnext)
        # wrong when it actualy spikes! doesn't matter for the use case
        S = jnp.where(ttfs == -1, superspike(v - self.vthres), v*0 + 1)
        return carry, (ttfs, S)

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

