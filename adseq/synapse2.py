import jax
import math
import jax.core
import jax.numpy as jnp
import typing
from .implementations import BaseQueue
from .synapse import apply_recv_gradient, time_to_timestep_keep_gradient, spike_detect

__all__ = 'mk_synapse2', 'mk_synapse2s'

floatx = jax.numpy.array(0.).dtype
def mk_synapse2(Q: type[BaseQueue], *a, dt_ms, vthres, tau_syn1_ms, tau_syn2_ms, **k):
    '''
    Construct a single double exponential synapse

    >>> syn = adseq.mk_synapse2(
            adseq.SingleSpike,
            dt_ms=1.0,
            vthres=1.0,
            tau_syn1_ms=1.0,
            tau_syn2_ms=10.0,
            max_delay_ms=100.,
            )
    >>> a = 1.0
    >>> syn = syn.timestep_spike_detect_pre(0, 1.0-a*0.1, 1.0+a*0.1, 1.0)
    >>> print(syn.isyn)
    0.0
    >>> syn = syn.timestep_spike_detect_pre(1, 0.9, 0.9, 0.0)
    >>> print(syn.isyn)
    0.0
    >>> syn = syn.timestep_spike_detect_pre(2, 0.9, 0.9, 0.0)
    >>> print(syn.isyn)
    0.7705642

    If taking gradient w.r.t. delay_ms, put max_delay_ms and give delay_ms in the detection call
    '''
    return _mk_synapse(Q, *a, dt_ms=dt_ms, vthres=vthres, tau_syn1_ms=tau_syn1_ms, tau_syn2_ms=tau_syn2_ms, **k).init()

def mk_synapse2s(Q: type[BaseQueue], *a, dt_ms, vthres, tau_syn1_ms, tau_syn2_ms, n: int, **k):
    return _mk_multi_synapse(Q, *a, dt_ms=dt_ms, vthres=vthres, tau_syn1_ms=tau_syn1_ms, tau_syn2_ms=tau_syn2_ms, n=n, **k).init()

###

def _mk_synapse(Q: type[BaseQueue], *a, max_delay_ms, dt_ms, vthres, tau_syn1_ms, tau_syn2_ms, **k):
    assert max_delay_ms is not None
    alpha = jnp.exp(- dt_ms / tau_syn1_ms)
    beta = jnp.exp(- dt_ms / tau_syn2_ms)
    t_peak = (tau_syn1_ms * tau_syn2_ms / (tau_syn2_ms - tau_syn1_ms) * jnp.log(tau_syn2_ms / tau_syn1_ms))
    denom = (jnp.exp(-t_peak / tau_syn2_ms) - jnp.exp(-t_peak / tau_syn1_ms))
    class StaticSynapse(typing.NamedTuple):
        queue: BaseQueue
        isyn1: float | jax.Array
        isyn2: float | jax.Array
        @property
        def isyn(self):
            return (self.isyn2-self.isyn1) / denom
        @classmethod
        def init(cls):
            return cls(Q.init(int(math.ceil(max_delay_ms/dt_ms)), *a, **k, grad=True), jnp.array(0.), jnp.array(0.)) # type: ignore
        def timestep_spike_detect_pre(self, t_ms, v, vnext, delay_ms):
            't_ms and delay_ms in ms, vnext is used for spike detection and dvdt estimation'
            tpost = spike_detect(dt_ms, t_ms, vthres, v, vnext, delay_ms)
            queue = self.queue
            enqueued = queue.enqueue(time_to_timestep_keep_gradient(tpost, dt_ms)) # type: ignore
            queue = jax.tree.map(lambda a, b: jnp.where(tpost != -1, a, b), enqueued, queue)
            queue, post_hit = queue.pop(time_to_timestep_keep_gradient(t_ms, dt_ms))
            isyn1 = alpha * self.isyn1 + apply_recv_gradient(post_hit, tau_syn1_ms)
            isyn2 = beta * self.isyn2 + apply_recv_gradient(post_hit, tau_syn2_ms)
            return StaticSynapse(queue, isyn1, isyn2)
    return StaticSynapse

def _mk_multi_synapse(Q: type[BaseQueue], *a, dt_ms, vthres, tau_syn1_ms, tau_syn2_ms, max_delay_ms=None, n, **k):
    # assert max_delay_ms is not None
    t_peak = (tau_syn1_ms * tau_syn2_ms / (tau_syn2_ms - tau_syn1_ms) * jnp.log(tau_syn2_ms / tau_syn1_ms))
    denom = (jnp.exp(-t_peak / tau_syn2_ms) - jnp.exp(-t_peak / tau_syn1_ms))
    alpha = jnp.exp(- dt_ms / tau_syn1_ms)
    beta = jnp.exp(- dt_ms / tau_syn2_ms)
    class StaticMultiSynapse(typing.NamedTuple):
        queues: BaseQueue
        isyn1: float | jax.Array
        isyn2: float | jax.Array
        @property
        def isyn(self):
            return (self.isyn2-self.isyn1) / denom
        @classmethod
        def init(cls):
            max_delay_steps = int(math.ceil(max_delay_ms/dt_ms)) if max_delay_ms is not None else None
            queues = jax.vmap(lambda _: Q.init(max_delay_steps, *a, **k, grad=True))(jnp.empty(n)) # type: ignore
            isyn1 = jnp.zeros(n, dtype=floatx)
            isyn2 = jnp.zeros(n, dtype=floatx)
            return cls(queues, isyn1, isyn2) # type: ignore
        def timestep_spike_detect_pre(self, t_ms, v, vnext, delay_ms):
            't_ms and delay_ms in ms, vnext is used for spike detection and dvdt estimation'
            assert len(delay_ms.shape) == 1 and delay_ms.shape[0] == n
            assert len(v.shape) == 1 and v.shape[0] == n
            assert len(vnext.shape) == 1 and vnext.shape[0] == n
            def timestep(queue, isyn1, isyn2, v, vnext, delay_ms):
                tpost = spike_detect(dt_ms, t_ms, vthres, v, vnext, delay_ms)
                enqueued = queue.enqueue(time_to_timestep_keep_gradient(tpost, dt_ms)) # type: ignore
                queue = jax.tree.map(lambda a, b: jnp.where(tpost != -1, a, b), enqueued, queue)
                queue, post_hit = queue.pop(time_to_timestep_keep_gradient(t_ms, dt_ms))
                isyn1 = alpha * isyn1 + apply_recv_gradient(post_hit, tau_syn1_ms)
                isyn2 = beta  * isyn2 + apply_recv_gradient(post_hit, tau_syn2_ms)
                return (queue, isyn1, isyn2)
            queues, isyn1, isyn2 = jax.vmap(timestep)(
                    self.queues, self.isyn1, self.isyn2,
                    v, vnext, delay_ms)
            return StaticMultiSynapse(queues, isyn1, isyn2)
        def timestep_static_spike(self, t_ms, s, delay_ms):
            assert len(delay_ms.shape) == 1 and delay_ms.shape[0] == n
            assert len(s.shape) == 1 and s.shape[0] == n
            def timestep(queue, isyn1, isyn2, s, delay_ms):
                tpost = t_ms + delay_ms
                enqueued = queue.enqueue(time_to_timestep_keep_gradient(tpost, dt_ms)) # type: ignore
                queue = jax.tree.map(lambda a, b: jnp.where(s, a, b), enqueued, queue)
                queue, post_hit = queue.pop(time_to_timestep_keep_gradient(t_ms, dt_ms))
                isyn1 = alpha * isyn1 + apply_recv_gradient(post_hit, tau_syn1_ms)
                isyn2 = beta  * isyn2 + apply_recv_gradient(post_hit, tau_syn2_ms)
                return (queue, isyn1, isyn2)
            queues, isyn1, isyn2 = jax.vmap(timestep)(
                    self.queues, self.isyn1, self.isyn2,
                    s, delay_ms)
            return StaticMultiSynapse(queues, isyn1, isyn2)
    return StaticMultiSynapse

