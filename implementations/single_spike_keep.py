import typing
import jax
import jax.numpy as jnp

INT_MAX = 0x7fffffff

__all__ = 'SingleSpikeKeep',

class SingleSpikeKeep(typing.NamedTuple):
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
def _enqueue(self: SingleSpikeKeep, n: float):
    empty = self.last_spike == INT_MAX
    return jax.lax.cond(empty,
                         lambda: SingleSpikeKeep(jnp.array(n, dtype=self.last_spike.dtype)),
                         lambda: self)
@_enqueue.defjvp
def _enqueue_grad(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    empty = self.last_spike == INT_MAX
    return jax.lax.cond(empty,
                         lambda: SingleSpikeKeep(jnp.array(n, dtype=self.last_spike.dtype)),
                         lambda: self), \
           jax.lax.cond(empty,
                         lambda: SingleSpikeKeep(jnp.array(n_t, dtype=self.last_spike.dtype)),
                         lambda: self_t)
del _enqueue_grad


@jax.custom_jvp
def _pop(self: SingleSpikeKeep, n: float):
    hit = self.last_spike <= n
    return (jax.lax.cond(hit,
                         lambda: SingleSpikeKeep(jnp.array(INT_MAX, dtype=self.last_spike.dtype)),
                         lambda: self),
            hit.astype(self.last_spike.dtype))
@_pop.defjvp
def _pop_grad(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    del n_t
    hit = self.last_spike <= n
    return (jax.lax.cond(hit,
                         lambda: SingleSpikeKeep(jnp.array(INT_MAX, dtype=self.last_spike.dtype)),
                         lambda: self),
            hit.astype(self.last_spike.dtype)), \
           (jax.lax.cond(hit,
                         lambda: SingleSpikeKeep(jnp.array(0, dtype=self.last_spike.dtype)),
                         lambda: self),
            self_t.last_spike)
del _pop_grad
