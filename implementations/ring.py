import jax
import jax.numpy as jnp
import typing

class Ring(typing.NamedTuple):
    buffer: jax.Array
    @classmethod
    def init(cls, delay):
        return cls(jnp.full(delay, 0, 'int32'))
    def enqueue(self, n):
        delay = self.buffer.shape[0]
        return Ring(self.buffer.at[(n+1) % delay].add(1))
    def pop(self, n):
        delay = self.buffer.shape[0]
        return Ring(self.buffer.at[n % delay].set(0)), \
                self.buffer[n % delay]

