import jax
import jax.numpy as jnp
import functools
import typing
from .ring import Ring

__all__ = 'LossyRing',

INT_MAX = 0x7fffffff

class LossyRing(typing.NamedTuple):
    buffer: jax.Array
    @classmethod
    def init(cls, delay, capacity):
        del delay
        return cls(jnp.full(capacity, INT_MAX, 'int32'))
    def enqueue(self, n):
        cap = self.buffer.shape[0]
        return LossyRing(self.buffer.at[n % cap].set(n))
    def pop(self, n):
        cap = self.buffer.shape[0]
        hit = self.buffer[n%cap] <= n
        return LossyRing(
                jax.lax.select(hit,
                    self.buffer.at[n % cap].set(INT_MAX),
                    self.buffer)), hit.astype(int)
    @classmethod
    def sized(cls, n):
        "wish I could use __class_getitem__"
        return type(f'{cls.__name__}[{n}]',
                    cls.__bases__,
                    {**cls.__dict__,
                     "init": functools.partial(cls.init, capacity=n)})
