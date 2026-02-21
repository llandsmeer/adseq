'''
Priority queue.
Every insert is a full memory move.
Better would be circular but then finding the insertion point is a bit harder
'''

import functools
import typing
import jax
import jax.numpy as jnp

INT_MAX = 0x7fffffff

__all__ = 'SortedArray',

class SortedArray(typing.NamedTuple):
    buffer: jax.Array
    @classmethod
    def init(cls, delay, capacity=None, grad=False):
        return cls(
                jnp.full(delay if capacity is None else capacity, INT_MAX, 'float32' if grad else 'int32'),
                )
    @classmethod
    def sized(cls, n):
        "wish I could use __class_getitem__"
        return type(f'{cls.__name__}[{n}]',
                    cls.__bases__,
                    {**cls.__dict__,
                     "init": functools.partial(cls.init, capacity=n)})
    def enqueue(self, n):
        return _enqueue(self, n)

    def pop(self, n):
        return _pop(self, n)

@jax.custom_jvp
def _enqueue(self, n):
    end = self.buffer.shape[0] - 1
    do_insert = self.buffer[end] > n
    return SortedArray(
       jax.lax.select(do_insert,
                      jnp.sort(self.buffer.at[end].set(n)),
                      self.buffer)
       )
@_enqueue.defjvp
def _enqueue_jvp(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    end = self.buffer.shape[0] - 1
    do_insert = self.buffer[end] > n
    buffer_next = self.buffer.at[end].set(n)
    buffer_next_t = self_t.buffer.at[end].set(n_t)
    order = jnp.argsort(buffer_next),
    return SortedArray(jax.lax.select(do_insert, buffer_next  [order], self.buffer)), \
           SortedArray(jax.lax.select(do_insert, buffer_next_t[order], self_t.buffer))
del _enqueue_jvp

@jax.custom_jvp
def _pop(self, n):
    hit = self.buffer[0] <= n
    return SortedArray(
       # should benchmark which one is better
       jax.lax.select(hit, jnp.roll(self.buffer.at[0].set(INT_MAX), -1), self.buffer)
       # jax.lax.select(hit, self.buffer.at[:-1].set(self.buffer[1:]).at[-1].set(INT_MAX), self.buffer)
       ), hit.astype(self.buffer.dtype)

@_pop.defjvp
def _pop_jvp(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    del n_t
    hit = self.buffer[0] <= n
    return \
        (SortedArray(
            jax.lax.select(hit, jnp.roll(self.buffer.at[0].set(INT_MAX), -1), self.buffer)
        ), hit.astype(self.buffer.dtype)), \
        (SortedArray(
            jax.lax.select(hit, jnp.roll(self_t.buffer.at[0].set(0), -1), self_t.buffer)
        ), self_t.buffer[0])
del _pop_jvp
