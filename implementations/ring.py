import jax
import jax.numpy as jnp
import typing

__all__ = 'Ring',

class Ring(typing.NamedTuple):
    buffer: jax.Array
    @classmethod
    def init(cls, delay, grad=False):
        return cls(jnp.full(
            delay+1,
            0,
            'float32' if grad else 'int32'))
    def enqueue(self, n):
        return _enqueue(self, n)
    def pop(self, n):
        return _pop(self, n)

@jax.custom_jvp
def _enqueue(self: Ring, n: float):
    delay = jnp.asarray(self.buffer.shape[0], dtype='int32')
    # THIS IS UGLY BUT GROQ DOES NOT SUPPORT DIVISION BY INT32 :(
    # NEED TO MAKE IT BACKEND SPECIFIC
    idx = jnp.asarray(n, dtype='int32') % delay.astype('float32')
    idx = idx.astype('int32')
    return Ring(self.buffer.at[idx].add(1, mode='promise_in_bounds'))
@_enqueue.defjvp
def _enqueue_grad(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    delay = jnp.asarray(self.buffer.shape[0], dtype='int32')
    # THIS IS UGLY BUT GROQ DOES NOT SUPPORT DIVISION BY INT32 :(
    # NEED TO MAKE IT BACKEND SPECIFIC
    idx = jnp.asarray(n, dtype='int32') % delay.astype('float32')
    idx = idx.astype('int32')
    return Ring(self.buffer.at[idx].add(1, mode='promise_in_bounds')), \
           Ring(self_t.buffer.at[idx].add(n_t, mode='promise_in_bounds'))
del _enqueue_grad

@jax.custom_jvp
def _pop(self, n):
    delay = jnp.asarray(self.buffer.shape[0], dtype='int32')
    # THIS IS UGLY BUT GROQ DOES NOT SUPPORT DIVISION BY INT32 :(
    # NEED TO MAKE IT BACKEND SPECIFIC
    idx = jnp.asarray(n, dtype='int32') % delay.astype('float32')
    idx = idx.astype('int32')
    return Ring(self.buffer.at[idx].set(0, mode='promise_in_bounds')), \
                self.buffer.at[idx].get(mode='promise_in_bounds')

@_pop.defjvp
def _pop_grad(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    del n_t
    delay = jnp.asarray(self.buffer.shape[0], dtype='int32')
    idx = jnp.asarray(n, dtype='int32') % delay.astype('float32')
    # THIS IS UGLY BUT GROQ DOES NOT SUPPORT DIVISION BY INT32 :(
    # NEED TO MAKE IT BACKEND SPECIFIC
    idx = idx.astype('int32')
    return (Ring(self.buffer.at[idx].set(0, mode='promise_in_bounds')),
                 self.buffer.at[idx].get(mode='promise_in_bounds')), \
           (Ring(self_t.buffer.at[idx].set(0, mode='promise_in_bounds')),
                 self_t.buffer.at[idx].get(mode='promise_in_bounds'))
del _pop_grad
