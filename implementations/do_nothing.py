import jax
import jax.numpy as jnp
import typing

__all__ = 'DoNothing',

class DoNothing(typing.NamedTuple):
    empty: jax.Array
    @classmethod
    def init(cls, delay):
        del delay
        return cls(jnp.array(0))
    def enqueue(self, n):
        del n
        return self
    def enqueue_with_value(self, n, v):
        del n, v
        return self
    def pop(self, n):
        del n
        return self, 0
