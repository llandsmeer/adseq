import jax
import jax.numpy as jnp
import typing
from implementations import BaseQueue

__all__ = 'mk_synapse', 'mk_synapses'

def mk_synapse(Q: BaseQueue, *a, delay_ms, dt_ms, vthres, tau_syn_ms, **k):
    return _mk_synapse(Q, *a, delay_ms=delay_ms, dt_ms=dt_ms, vthres=vthres, tau_syn_ms=tau_syn_ms, **k).init()

def mk_synapses(Q: BaseQueue, *a, delay_ms, dt_ms, vthres, tau_syn_ms, n:int, **k):
    init = _mk_synapse(Q, *a, delay_ms=delay_ms, dt_ms=dt_ms, vthres=vthres, tau_syn_ms=tau_syn_ms, **k).init()
    return jax.vmap(lambda _: init)(jnp.arange(n))

###

def _mk_synapse(Q: BaseQueue, *a, delay_ms, dt_ms, vthres, tau_syn_ms, **k):
    alpha = jnp.exp(- dt_ms / tau_syn_ms)
    class StaticSynapse(typing.NamedTuple):
        queue: BaseQueue
        isyn: float | jax.Array
        @classmethod
        def init(cls):
            return cls(Q.init(int(jnp.ceil(delay_ms/dt_ms)), *a, **k, grad=True), jnp.array(0.)) # type: ignore
        def timestep_spike_detect_pre(self, ts, v, vnext):
            tpre = spike_detect(dt_ms, ts, vthres, v, vnext)
            tpost = tpre + delay_ms
            queue = self.queue
            queue = jax.lax.cond(tpre != -1, # must be a better solution
                 lambda: queue.enqueue(time_to_timestep_keep_gradient(tpost, dt_ms)), # type: ignore
                 lambda: queue)
            queue, post_hit = queue.pop(time_to_timestep_keep_gradient(ts, dt_ms))
            isyn = alpha * self.isyn + apply_recv_gradient(post_hit, tau_syn_ms)
            return StaticSynapse(queue, isyn)
    return StaticSynapse

@jax.custom_jvp
def spike_detect(dt, ts, vthres, v, vnext):
    del dt
    hit = (v < vthres) & (vnext >= vthres)
    return jax.lax.select(hit, ts, -1.)

@spike_detect.defjvp
def spike_detect_vjp(primals, tangents):
    dt, ts, vthres, v, vnext = primals
    _, _, _, v_dot, vnext_dot = tangents
    dvdt = (vnext - v) / dt
    hit = (v < vthres) & (vnext >= vthres)
    primal_out = jax.lax.select(hit, ts, -1.)
    tangent_out = jax.lax.select(hit, - 1/dvdt * v_dot, 0.)
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
    return hit, -1/tau_syn * tpost_t
del apply_recv_gradient_jvp
