import jax
import math
import jax.core
import jax.numpy as jnp
import typing
from .implementations import BaseQueue

__all__ = 'mk_synapse2', 'mk_synapse2s'

floatx = jax.numpy.array(0.).dtype
def mk_synapse2(Q: BaseQueue, *a, dt_ms, vthres, tau_syn1_ms, tau_syn2_ms, **k):
    return _mk_synapse(Q, *a, dt_ms=dt_ms, vthres=vthres, tau_syn1_ms=tau_syn1_ms, tau_syn2_ms=tau_syn2_ms, **k).init()

def mk_synapse2s(Q: BaseQueue, *a, dt_ms, vthres, tau_syn1_ms, tau_syn2_ms, n: int, **k):
    return _mk_multi_synapse(Q, *a, dt_ms=dt_ms, vthres=vthres, tau_syn1_ms=tau_syn1_ms, tau_syn2_ms=tau_syn2_ms, n=n, **k).init()

###

def _mk_synapse(Q: type[BaseQueue], *a, max_delay_ms, dt_ms, vthres, tau_syn1_ms, tau_syn2_ms, **k):
    '''
    If taking gradient w.r.t. delay_ms, put max_delay_ms
    '''
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
        def timestep_spike_detect_pre(self, ts, v, vnext, delay_ms):
            tpost = spike_detect(dt_ms, ts, vthres, v, vnext, delay_ms)
            queue = self.queue
            queue = jax.lax.cond(tpost != -1, # must be a better solution
                 lambda: queue.enqueue(time_to_timestep_keep_gradient(tpost, dt_ms)), # type: ignore
                 lambda: queue)
            queue, post_hit = queue.pop(time_to_timestep_keep_gradient(ts, dt_ms))
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
            return cls(queues, isyn, isyn) # type: ignore
        def timestep_spike_detect_pre(self, ts, v, vnext, delay_ms):
            assert len(delay_ms.shape) == 1 and delay_ms.shape[0] == n
            assert len(v.shape) == 1 and v.shape[0] == n
            assert len(vnext.shape) == 1 and vnext.shape[0] == n
            def timestep(queue, isyn1, isyn2, v, vnext, delay_ms):
                tpost = spike_detect(dt_ms, ts, vthres, v, vnext, delay_ms)
                queue = jax.lax.cond(tpost != -1, # must be a better solution
                     lambda: queue.enqueue(time_to_timestep_keep_gradient(tpost, dt_ms)), # type: ignore
                     lambda: queue)
                queue, post_hit = queue.pop(time_to_timestep_keep_gradient(ts, dt_ms))
                isyn1 = alpha * isyn1 + apply_recv_gradient(post_hit, tau_syn1_ms)
                isyn2 = beta  * isyn2 + apply_recv_gradient(post_hit, tau_syn2_ms)
                return (queue, isyn1, isyn2)
            queues, isyn1, isyn2 = jax.vmap(timestep)(
                    self.queues, self.isyn1, self.isyn2,
                    v, vnext, delay_ms)
            return StaticMultiSynapse(queues, isyn1, isyn2)
        def timestep_static_spike(self, ts, s, delay_ms):
            assert len(delay_ms.shape) == 1 and delay_ms.shape[0] == n
            assert len(s.shape) == 1 and s.shape[0] == n
            def timestep(queue, isyn1, isyn2, s, delay_ms):
                tpost = ts + delay_ms
                queue = jax.lax.cond(s, # must be a better solution
                     lambda: queue.enqueue(time_to_timestep_keep_gradient(tpost, dt_ms)), # type: ignore
                     lambda: queue)
                queue, post_hit = queue.pop(time_to_timestep_keep_gradient(ts, dt_ms))
                isyn1 = alpha * isyn1 + apply_recv_gradient(post_hit, tau_syn1_ms)
                isyn2 = beta  * isyn2 + apply_recv_gradient(post_hit, tau_syn2_ms)
                return (queue, isyn1, isyn2)
            queues, isyn1, isyn2 = jax.vmap(timestep)(
                    self.queues, self.isyn1, self.isyn2,
                    s, delay_ms)
            return StaticMultiSynapse(queues, isyn1, isyn2)
    return StaticMultiSynapse

@jax.custom_jvp
def spike_detect(dt, ts, vthres, v, vnext, delay):
    del dt
    hit = (v < vthres) & (vnext >= vthres)
    return jax.lax.select(hit, ts + delay, -1.)

@spike_detect.defjvp
def spike_detect_vjp(primals, tangents):
    dt, ts, vthres, v, vnext, delay = primals
    _, _, _, v_dot, vnext_dot, delay_t = tangents
    del vnext_dot
    dvdt = (vnext - v) / dt
    hit = (v < vthres) & (vnext >= vthres)
    primal_out = jax.lax.select(hit, ts + delay, -1.)
    dvdt_safe = jnp.where(dvdt == 0, 1, dvdt) # wrong but no nans
    tangent_out = jax.lax.select(hit, - 1/dvdt_safe * v_dot + delay_t, 0.)
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
    return hit
@apply_recv_gradient.defjvp
def apply_recv_gradient_jvp(primals, tangents):
    hit, tau_syn = primals
    tpost_t, tau_syn_t = tangents
    del tau_syn_t
    return hit, hit * (1/tau_syn * tpost_t)
del apply_recv_gradient_jvp
