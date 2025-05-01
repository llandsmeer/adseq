'''
Assumes sorted inputs, ie homogeneous delays
'''

import functools
import typing
import jax
import jax.numpy as jnp

INT_MAX = 0x7fffffff

__all__ = 'FIFORing',

class FIFORing(typing.NamedTuple):
    buffer: jax.Array
    head: int | jax.Array
    size: int | jax.Array
    @classmethod
    def init(cls, delay, capacity=None):
        return cls(
                jnp.full(delay if capacity is None else capacity, INT_MAX, 'int32'),
                0, 0
                )
    def enqueue(self, n):
        cap = self.buffer.shape[0]
        do_insert = self.size < cap
        return FIFORing(
           jax.lax.select(do_insert, self.buffer.at[(self.head + self.size) % cap].set(n), self.buffer),
           self.head,
           jax.lax.select(do_insert, self.size+1, self.size)
           )
    def pop(self, n):
        cap = self.buffer.shape[0]
        hit = self.buffer[self.head] <= n
        return FIFORing(
           jax.lax.select(hit, self.buffer.at[self.head].set(INT_MAX), self.buffer),
           jax.lax.select(hit, (self.head+1) % cap, self.head),
           jax.lax.select(hit, self.size-1, self.size)
           ), hit.astype(int)
    @classmethod
    def sized(cls, n):
        "wish I could use __class_getitem__"
        return type(f'{cls.__name__}[{n}]',
                    cls.__bases__,
                    {**cls.__dict__,
                     "init": functools.partial(cls.init, capacity=n)})

