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
    def init(cls, delay, capacity=None):
        return cls(
                jnp.full(delay if capacity is None else capacity, INT_MAX, 'int32'),
                )
    def enqueue(self, n):
        end = self.buffer.shape[0] - 1
        do_insert = self.buffer[end] > n
        return SortedArray(
           jax.lax.select(do_insert,
                          jnp.sort(self.buffer.at[end].set(n)),
                          self.buffer)
           )
    def pop(self, n):
        hit = self.buffer[0] <= n
        return SortedArray(
           jax.lax.select(hit, self.buffer.at[0].set(INT_MAX), self.buffer)
           ), hit.astype(int)
    @classmethod
    def sized(cls, n):
        "wish I could use __class_getitem__"
        return type(f'{cls.__name__}[{n}]',
                    cls.__bases__,
                    {**cls.__dict__,
                     "init": functools.partial(cls.init, capacity=n)})

