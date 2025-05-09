import jax
import jax.numpy as jnp
import typing

__all__ = 'Ring',

class Ring(typing.NamedTuple):
    buffer: jax.Array
    @classmethod
    def init(cls, delay):
        return cls(jnp.full(delay, 0, 'int32'))
    def enqueue(self, n):
        delay = self.buffer.shape[0]
        return Ring(self.buffer.at[n % delay].add(1))
    def enqueue_with_value(self, n, v):
        delay = self.buffer.shape[0]
        return Ring(self.buffer.at[n % delay].set(v))
    def pop(self, n):
        delay = self.buffer.shape[0]
        return Ring(self.buffer.at[n % delay].set(0)), \
                self.buffer[n % delay]

