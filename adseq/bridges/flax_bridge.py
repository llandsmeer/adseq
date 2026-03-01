import flax.linen as nn
import abc

import typing
import jax

from .. import implementations
from .. import synapse2

class StaticMultiSynapse(abc.ABC):
    @property
    @abc.abstractmethod
    def isyn(self) -> float: ...

    @abc.abstractmethod
    def timestep_spike_detect_pre(self, ts, v, vnext, delay_ms) -> typing.Self: ... # type: ignore

    @abc.abstractmethod
    def timestep_static_spike(self, ts, s, delay_ms) -> typing.Self: ... # type: ignore

class DelaySynapse(nn.Module):
    dt: float
    tau_syn1_ms: float = 0.5
    tau_syn2_ms: float = 2.0
    max_delay: float = 200.
    vthres: float = 1.0
    delay_init = nn.initializers.normal()
    delay_activation = lambda self, x: self.max_delay * (1 + nn.tanh(x))
    queue: type[implementations.BaseQueue] = implementations.FIFORing.sized(4) # type: ignore

    def init_carry(self, v) -> StaticMultiSynapse:
        assert len(v.shape) == 1
        syn = synapse2.mk_synapse2s(self.queue,
                                   vthres=self.vthres,
                                   tau_syn1_ms=self.tau_syn1_ms,
                                   tau_syn2_ms=self.tau_syn2_ms,
                                   dt_ms=self.dt,
                                   n=len(v),
                max_delay_ms=self.max_delay)
        return syn, 0 # type: ignore

    @nn.compact
    def __call__(self, carry: tuple[StaticMultiSynapse, int], v: jax.Array, vnext: jax.Array|None):
        delay = self.delay_activation(self.param('delay', self.delay_init, v.shape))
        syn, ts = carry
        isyn = syn.isyn
        syn = syn.timestep_spike_detect_pre(ts=ts, v=v, vnext=vnext, delay_ms=delay)
        return (syn, ts+1), isyn

