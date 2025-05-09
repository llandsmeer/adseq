import jax
import typing
from implementations import BaseQueue

__all__ = 'GradientQueue',

@jax.custom_jvp
def spiketime_detect(ts, v, dvdt, vnext=None, dt=None, vthres=1):
    if vnext is None:
        vnext = v + dvdt * dt
    hit = (v < vthres) & (vnext>= vthres)
    return jax.lax.select(hit, ts, float('inf')), hit

@spiketime_detect.defjvp
def spiketime_detect_vjp(primals, tangents):
    ts, v,     dvdt, vnext, dt, vthres = primals
    _,  v_dot, _,    _,     _,  _      = tangents
    if vnext is None:
        vnext = v + dvdt * dt
    hit = (v < vthres) & (vnext>= vthres)
    primal_out = jax.lax.select(hit, ts, float('inf'))
    tangent_out = jax.lax.select(hit, - 1/dvdt * v_dot, 0.)
    return primal_out, tangent_out

class GradientQueue:
    def __new__(cls, TQPrim: BaseQueue, TQTan: BaseQueue | None):
        if TQTan is None: TQTan = TQPrim
        @jax.custom_jvp
        def init(delay):
            return TQPrim.init(delay)
        @init.defjvp
        def init_jvp(p, t):
            delay, = p
            del t
            return TQPrim.init(delay), TQTan.init(delay)
        @jax.custom_jvp
        def enqueue(queue: BaseQueue, n):
            return queue.enqueue(n)
        @enqueue.defjvp
        def enqueue_jvp(p, t):
            queue_p, n, = p
            queue_t, n_t, = t
            return queue_p.enqueue(n), queue_t.enqueue_with_value(n, n_t)
        @jax.custom_vjp
        def pop(queue, n):
            return queue.pop(n)
        @pop.defjvp
        def pop_jvp(p, t):
            queue_p, n, = p
            queue_t, _, = t
            return queue_p.pop(n), (queue_p.pop(n) != 0) * -queue_t.pop(n)
        del init_jvp, enqueue_jvp, pop_jvp
        class GradQueue(typing.NamedTuple):
            queue: BaseQueue
            @classmethod
            def init(cls, delay): return GradQueue(init(delay))
            def enqueue(self, n): return GradQueue(enqueue(self.queue, n))
            def pop(self, n):     return GradQueue(pop(self.queue, n))
        return GradQueue
    def __class_getitem__(cls, TQPrim: BaseQueue, TQTan: BaseQueue | None):
        return cls.__new__(cls, TQPrim, TQTan)
