import jax
import jax.numpy as jnp
import functools
import typing

__all__ = 'LossyRing',

INT_MAX = 0x7fffffff

class LossyRing(typing.NamedTuple):
    buffer: jax.Array
    @classmethod
    def init(cls, delay, capacity, grad=False):
        del delay
        return cls(jnp.full(capacity, INT_MAX,
            'float32' if grad else 'int32'))
    def enqueue(self, n):
        return _enqueue(self, n)
    def pop(self, n):
        return _pop(self, n)
    @classmethod
    def sized(cls, n):
        "wish I could use __class_getitem__"
        return type(f'{cls.__name__}[{n}]',
                    cls.__bases__,
                    {**cls.__dict__,
                     "init": functools.partial(cls.init, capacity=n)})

@jax.custom_jvp
def _enqueue(self: LossyRing, n: float):
    capacity = jnp.asarray(self.buffer.shape[0], dtype='int32')
    # XXX: THIS IS UGLY BUT GROQ DOES NOT SUPPORT DIVISION BY INT32 :(
    # NEED TO MAKE IT BACKEND SPECIFIC
    idx = jnp.asarray(n, dtype='int32') % capacity.astype('float32')
    idx = idx.astype('int32')
    return LossyRing(self.buffer.at[idx].set(n, mode='promise_in_bounds'))
@_enqueue.defjvp
def _enqueue_grad(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    capacity = jnp.asarray(self.buffer.shape[0], dtype='int32')
    # XXX: THIS IS UGLY BUT GROQ DOES NOT SUPPORT DIVISION BY INT32 :(
    # NEED TO MAKE IT BACKEND SPECIFIC
    idx = jnp.asarray(n, dtype='int32') % capacity.astype('float32')
    idx = idx.astype('int32')
    return LossyRing(self.buffer.at[idx].set(n, mode='promise_in_bounds')), \
           LossyRing(self_t.buffer.at[idx].set(n_t, mode='promise_in_bounds'))
del _enqueue_grad

@jax.custom_jvp
def _pop(self, n):
    capacity = jnp.asarray(self.buffer.shape[0], dtype='int32')
    # XXX: THIS IS UGLY BUT GROQ DOES NOT SUPPORT DIVISION BY INT32 :(
    # NEED TO MAKE IT BACKEND SPECIFIC
    idx = jnp.asarray(n, dtype='int32') % capacity.astype('float32')
    idx = idx.astype('int32')
    root = self.buffer.at[idx].get(mode='promise_in_bounds')
    hit = root <= n
    return LossyRing(
            jax.lax.select(hit,
                self.buffer.at[idx].set(INT_MAX, mode='promise_in_bounds'),
                self.buffer)), \
                hit.astype(self.buffer.dtype)

@_pop.defjvp
def _pop_grad(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    del n_t
    capacity = jnp.asarray(self.buffer.shape[0], dtype='int32')
    # XXX: THIS IS UGLY BUT GROQ DOES NOT SUPPORT DIVISION BY INT32 :(
    # NEED TO MAKE IT BACKEND SPECIFIC
    idx = jnp.asarray(n, dtype='int32') % capacity.astype('float32')
    idx = idx.astype('int32')
    root = self.buffer.at[idx].get(mode='promise_in_bounds')
    hit = root <= n
    primal_out = LossyRing(
            jax.lax.select(hit,
                self.buffer.at[idx].set(INT_MAX, mode='promise_in_bounds'),
                self.buffer)), \
                hit.astype(self.buffer.dtype)
    tangent_out = LossyRing(
            jax.lax.select(hit,
                self_t.buffer.at[idx].set(0, mode='promise_in_bounds'),
                self_t.buffer)), \
                self_t.buffer.at[idx].get(mode='promise_in_bounds')
    return primal_out, tangent_out
del _pop_grad
