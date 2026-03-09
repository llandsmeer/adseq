import typing
import jax
import jax.numpy as jnp

INT_MAX = 0x7fffffff

__all__ = 'SingleSpike',

class SingleSpike(typing.NamedTuple):
    last_spike: jax.Array
    @classmethod
    def init(cls, delay, grad=False):
        del delay
        return cls(jnp.array(INT_MAX if not grad else float(INT_MAX)))
    def enqueue(self, n):
        return _enqueue(self, n)
    def pop(self, n):
        return _pop(self, n)

@jax.custom_jvp
def _enqueue(self: SingleSpike, n: float):
    return SingleSpike(jnp.array(n, dtype=self.last_spike.dtype))
@_enqueue.defjvp
def _enqueue_grad(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    return _enqueue(self, n), SingleSpike(jnp.array(n_t, dtype=self.last_spike.dtype))

del _enqueue_grad


@jax.custom_jvp
def _pop(self: SingleSpike, n: float):
    hit = self.last_spike <= n
    return (jax.lax.cond(hit,
                         lambda: SingleSpike(jnp.array(INT_MAX, dtype=self.last_spike.dtype)),
                         lambda: self),
            hit.astype(self.last_spike.dtype))
@_pop.defjvp
def _pop_grad(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    del n_t
    hit = self.last_spike <= n
    return (jax.lax.cond(hit,
                         lambda: SingleSpike(jnp.array(INT_MAX, dtype=self.last_spike.dtype)),
                         lambda: self),
            hit.astype(self.last_spike.dtype)), \
           (jax.lax.cond(hit,
                         lambda: SingleSpike(jnp.array(0, dtype=self.last_spike.dtype)),
                         lambda: self_t),
            self_t.last_spike)
del _pop_grad

### @jax.custom_vjp
### def _enqueue(self: SingleSpike, n: float):
###     return SingleSpike(jnp.array(n, dtype=self.last_spike.dtype))
### def _enqueue_fwd(self: SingleSpike, n: float):
###     out = SingleSpike(jnp.array(n, dtype=self.last_spike.dtype))
###     return out, None
### def _enqueue_bwd(res, g):
###     return (
###         SingleSpike(jnp.array(0, dtype=g.last_spike.dtype)),
###         g.last_spike
###     )
### _enqueue.defvjp(_enqueue_fwd, _enqueue_bwd)

### @jax.custom_vjp
### def _pop(self: SingleSpike, n: float):
###     hit = self.last_spike <= n
###     return (
###         jax.lax.cond(
###             hit,
###             lambda: SingleSpike(jnp.array(INT_MAX, dtype=self.last_spike.dtype)),
###             lambda: self,
###         ),
###         hit.astype(self.last_spike.dtype),
###     )
### 
### def _pop_fwd(self: SingleSpike, n: float):
###     hit = self.last_spike <= n
###     out = (
###         jax.lax.cond(
###             hit,
###             lambda: SingleSpike(jnp.array(INT_MAX, dtype=self.last_spike.dtype)),
###             lambda: self,
###         ),
###         hit.astype(self.last_spike.dtype),
###     )
###     return out, (hit,)
### 
### 
### def _pop_bwd(res, g):
###     (hit,) = res
###     g_state, g_hit = g
###     dtype = g_state.last_spike.dtype
###     d_self = SingleSpike(
###         jnp.where(hit, 0, g_state.last_spike)
###     )
###     d_n = jnp.array(0, dtype=dtype)
###     return d_self, d_n
### 
### _pop.defvjp(_pop_fwd, _pop_bwd)
