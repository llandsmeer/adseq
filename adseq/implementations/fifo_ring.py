'''
Assumes sorted inputs, ie homogeneous delays
'''

import functools
import typing
import jax
import jax.numpy as jnp

INT_MAX = 0x7fffffff

__all__ = 'FIFORing',

floatx = jax.numpy.array(0.).dtype

class FIFORing(typing.NamedTuple):
    buffer: jax.Array
    head: int | jax.Array
    size: int | jax.Array
    @classmethod
    def init(cls, delay, capacity=None, grad=False):
        return cls(
                jnp.full(delay if capacity is None else capacity, INT_MAX, floatx if grad else 'int32'),
                0, 0
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
    cap = self.buffer.shape[0]
    do_insert = self.size < cap
    return FIFORing(
       jax.lax.select(do_insert, self.buffer.at[(self.head + self.size) % cap].set(n, mode='promise_in_bounds'), self.buffer),
       self.head,
       jax.lax.select(do_insert, self.size+1, self.size)
       )
@_enqueue.defjvp
def _enqueue_jvp(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    cap = self.buffer.shape[0]
    do_insert = self.size < cap
    return FIFORing(
               jax.lax.select(do_insert, self.buffer.at[(self.head + self.size) % cap].set(n, mode='promise_in_bounds'), self.buffer),
               self.head,
               jax.lax.select(do_insert, self.size+1, self.size)
       ),  FIFORing(
               jax.lax.select(do_insert, self_t.buffer.at[(self.head + self.size) % cap].set(n_t, mode='promise_in_bounds'), self_t.buffer),
               self_t.head,
               self_t.size
       )
del _enqueue_jvp

@jax.custom_jvp
def _pop(self, n):
    cap = self.buffer.shape[0]
    hit = self.buffer.at[self.head].get(mode='promise_in_bounds') <= n
    return FIFORing(
       jax.lax.select(hit, self.buffer.at[self.head].set(INT_MAX, mode='promise_in_bounds'), self.buffer),
       jax.lax.select(hit, (self.head+1) % cap, self.head),
       jax.lax.select(hit, self.size-1, self.size)
       ), hit.astype(self.buffer.dtype)

@_pop.defjvp
def _pop_jvp(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    del n_t
    cap = self.buffer.shape[0]
    hit = self.buffer.at[self.head].get(mode='promise_in_bounds') <= n
    return (FIFORing(
               jax.lax.select(hit, self.buffer.at[self.head].set(INT_MAX, mode='promise_in_bounds'), self.buffer),
               jax.lax.select(hit, (self.head+1) % cap, self.head),
               jax.lax.select(hit, self.size-1, self.size)
       ), hit.astype(self.buffer.dtype)), (
           FIFORing(
               jax.lax.select(hit, self_t.buffer.at[self.head].set(INT_MAX, mode='promise_in_bounds'), self_t.buffer),
               self_t.head,
               self_t.size
       ), self_t.buffer[self.head])
del _pop_jvp
