import jax
import math
import jax.core
import jax.numpy as jnp
import typing
from .implementations import BaseQueue

__all__ = 'mk_synapse', 'mk_synapses'

floatx = jax.numpy.array(0.).dtype

def mk_synapse(Q: type[BaseQueue], *a, delay_ms, dt_ms, vthres, tau_syn_ms, **k):
    '''
    Construct a single simple exponential synapse

    >>> syn = adseq.mk_synapse(
    ...     adseq.SingleSpike,
    ...     dt_ms=0.1,
    ...     vthres=1.0,
    ...     tau_syn_ms=1.0,
    ...     delay_ms=100.
    ...     )
    >>> a = 1.0
    >>> syn = syn.timestep_spike_detect_pre(0, 1.0-a*0.1, 1.0+a*0.1, 1.0)
    >>> print(syn.isyn)
    0.0
    >>> syn = syn.timestep_spike_detect_pre(1, 0.9, 0.9, 0.0)
    >>> print(syn.isyn)
    1.0
    >>> syn = syn.timestep_spike_detect_pre(2, 0.9, 0.9, 0.0)
    >>> print(syn.isyn)
    0.9048374
    '''
    return _mk_synapse(Q, *a, delay_ms=delay_ms, dt_ms=dt_ms, vthres=vthres, tau_syn_ms=tau_syn_ms, **k).init()

def mk_synapses(Q: type[BaseQueue], *a, delay_ms, dt_ms, vthres, tau_syn_ms, n: int, **k):
    return _mk_multi_synapse(Q, *a, delay_ms=delay_ms, dt_ms=dt_ms, vthres=vthres, tau_syn_ms=tau_syn_ms, n=n, **k).init()

###

def _mk_synapse(Q: type[BaseQueue], *a, delay_ms, dt_ms, vthres, tau_syn_ms, max_delay_ms=None, **k):
    '''
    If taking gradient w.r.t. delay_ms, put max_delay_ms
    '''
    delay_ms_is_concrete = not isinstance(delay_ms, jax.core.Tracer)
    max_delay_ms = delay_ms if max_delay_ms is None and delay_ms_is_concrete else max_delay_ms
    assert max_delay_ms is not None
    delay_ms = jnp.asarray(delay_ms, dtype=floatx)
    alpha = jnp.exp(- dt_ms / tau_syn_ms)
    class StaticSynapse(typing.NamedTuple):
        queue: BaseQueue
        isyn: float | jax.Array
        @classmethod
        def init(cls):
            return cls(Q.init(int(math.ceil(max_delay_ms/dt_ms)), *a, **k, grad=True), jnp.array(0.)) # type: ignore
        def timestep_spike_detect_pre(self, ts, v, vnext, delay_ms=delay_ms):
            tpost = spike_detect(dt_ms, ts, vthres, v, vnext, delay_ms)
            queue = self.queue
            enqueued = queue.enqueue(time_to_timestep_keep_gradient(tpost, dt_ms)) # type: ignore
            queue = jax.tree.map(lambda a, b: jnp.where(tpost != -1, a, b), enqueued, queue)
            queue, post_hit = queue.pop(time_to_timestep_keep_gradient(ts, dt_ms))
            isyn = alpha * self.isyn + apply_recv_gradient(post_hit, tau_syn_ms)
            return StaticSynapse(queue, isyn)
    return StaticSynapse

def _mk_multi_synapse(Q: type[BaseQueue], *a, delay_ms, dt_ms, vthres, tau_syn_ms, max_delay_ms=None, n, **k):
    delay_ms_is_concrete = not isinstance(delay_ms, jax.core.Tracer) and (not hasattr(delay_ms, 'shape') or len(delay_ms) == 1)
    max_delay_ms = delay_ms if max_delay_ms is None and delay_ms_is_concrete else max_delay_ms
    # assert max_delay_ms is not None
    delay_ms = jnp.asarray(delay_ms, dtype=floatx)
    alpha = jnp.exp(- dt_ms / tau_syn_ms)
    class StaticMultiSynapse(typing.NamedTuple):
        queues: BaseQueue
        isyn: float | jax.Array
        @classmethod
        def init(cls):
            max_delay_steps = int(math.ceil(max_delay_ms/dt_ms)) if max_delay_ms is not None else None
            queues = jax.vmap(lambda _: Q.init(max_delay_steps, *a, **k, grad=True))(jnp.empty(n)) # type: ignore
            isyn = jnp.zeros(n, dtype=floatx)
            return cls(queues, isyn) # type: ignore
        def timestep_spike_detect_pre(self, ts, v, vnext, delay_ms=delay_ms):
            if len(delay_ms.shape) == 0 or delay_ms.shape[0] == 1:
                delay_ms = jnp.full(n, delay_ms)
            assert len(delay_ms.shape) == 1 and delay_ms.shape[0] == n
            assert len(v.shape) == 1 and v.shape[0] == n
            assert len(vnext.shape) == 1 and vnext.shape[0] == n
            def timestep(queue, isyn, v, vnext, delay_ms):
                tpost = spike_detect(dt_ms, ts, vthres, v, vnext, delay_ms)
                enqueued = queue.enqueue(time_to_timestep_keep_gradient(tpost, dt_ms)) # type: ignore
                queue = jax.tree.map(lambda a, b: jnp.where(tpost != -1, a, b), enqueued, queue)
                queue, post_hit = queue.pop(time_to_timestep_keep_gradient(ts, dt_ms))
                isyn = alpha * isyn + \
                       apply_recv_gradient(post_hit, tau_syn_ms)
                return (queue, isyn)
            queues, isyn = jax.vmap(timestep)(
                    self.queues, self.isyn,
                    v, vnext, delay_ms)
            return StaticMultiSynapse(queues, isyn)
        def timestep_static_spike(self, ts, s, delay_ms=delay_ms):
            if len(delay_ms.shape) == 0 or delay_ms.shape[0] == 1:
                delay_ms = jnp.full(n, delay_ms)
            assert len(delay_ms.shape) == 1 and delay_ms.shape[0] == n
            assert len(s.shape) == 1 and s.shape[0] == n
            def timestep(queue, isyn, s, delay_ms):
                tpost = ts + delay_ms
                enqueued = queue.enqueue(time_to_timestep_keep_gradient(tpost, dt_ms)) # type: ignore
                queue = jax.tree.map(lambda a, b: jnp.where(s, a, b), enqueued, queue)
                queue, post_hit = queue.pop(time_to_timestep_keep_gradient(ts, dt_ms))
                isyn = alpha * isyn + \
                       apply_recv_gradient(post_hit, tau_syn_ms)
                return (queue, isyn)
            queues, isyn = jax.vmap(timestep)(
                    self.queues, self.isyn,
                    s, delay_ms)
            return StaticMultiSynapse(queues, isyn)
    return StaticMultiSynapse

@jax.custom_jvp
def spike_detect(dt, t, vthres, v, vnext, delay):
    hit = (v < vthres) & (vnext >= vthres)
    del dt
    return jax.lax.select(hit, t + delay, -1.)

@spike_detect.defjvp
def spike_detect_vjp(primals, tangents):
    dt, t, vthres, v, vnext, delay = primals
    _, _, _, v_t, vnext_t, delay_t = tangents
    del vnext_t
    dvdt = (vnext - v) / dt
    hit = (v < vthres) & (vnext >= vthres)
    primal_out = jax.lax.select(hit, t + delay, -1.)
    dvdt_safe = jnp.where(dvdt == 0, 1, dvdt) # wrong but no nans and can't happen in case hit==1
    tangent_out = jax.lax.select(hit, - 1/dvdt_safe * v_t + delay_t, 0.)
    return primal_out, tangent_out


@jax.custom_jvp
def time_to_timestep_keep_gradient(x, dt):
    return jnp.round(x/dt)
@time_to_timestep_keep_gradient.defjvp
def time_to_timestep_keep_gradient_jvp(primals, tangents):
    x, dt = primals
    x_t, dt_t = tangents
    del dt_t
    return jnp.round(x/dt), x_t

@jax.custom_jvp
def apply_recv_gradient(hit, tau_syn):
    del tau_syn
    return jax.lax.select(hit != 0, 1.0, 0.0)
@apply_recv_gradient.defjvp
def apply_recv_gradient_jvp(primals, tangents):
    hit, tau_syn = primals
    tpost_t, tau_syn_t = tangents
    del tau_syn_t
    return jax.lax.select(hit != 0, 1.0, 0.0), \
           jax.lax.select(hit != 0, (1/tau_syn * tpost_t), 0.0)
del apply_recv_gradient_jvp
