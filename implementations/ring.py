import jax
import jax.numpy as jnp
import typing

__all__ = 'Ring',


class Ring(typing.NamedTuple):
    buffer: jax.Array
    @classmethod
    def init(cls, delay, grad=False):
        return cls(jnp.full(delay, 0, 'float32' if grad else 'int32'))
    def enqueue(self, n):
        return _enqueue(self, n)
    def pop(self, n):
        return _pop(self, n)

@jax.custom_jvp
def _enqueue(self: Ring, n: float):
    delay = self.buffer.shape[0]
    idx = jnp.asarray(n, dtype=int) % delay
    return Ring(self.buffer.at[idx].add(1))
@_enqueue.defjvp
def _enqueue_grad(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    delay = self.buffer.shape[0]
    idx = jnp.asarray(n, dtype=int) % delay
    return _enqueue(self, n), Ring(self_t.buffer.at[idx].add(n_t))
del _enqueue_grad

@jax.custom_jvp
def _pop(self, n):
    delay = self.buffer.shape[0]
    idx = jnp.asarray(n, dtype=int) % delay
    return Ring(self.buffer.at[idx].set(0)), \
            self.buffer[idx]

@_pop.defjvp
def _pop_grad(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    del n_t
    delay = self.buffer.shape[0]
    idx = jnp.asarray(n, dtype=int) % delay
    return (Ring(self.buffer.at[idx].set(0)),
            self.buffer[idx]), \
           (Ring(self_t.buffer.at[idx].set(0)),
            self_t.buffer[idx])
del _pop_grad
