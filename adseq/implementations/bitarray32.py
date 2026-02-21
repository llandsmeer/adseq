import jax
import jax.numpy as jnp
import typing

__all__ = 'BitArray32',

class BitArray32(typing.NamedTuple):
    buffer: jax.Array | int
    @classmethod
    def init(cls, delay=None):
        if not (delay is None or delay < 32):
            print('Warning: BitArray32 can only handle delay < 32')
        return cls(jnp.zeros(shape=(), dtype='int32'))
    def enqueue(self, n):
        return BitArray32(self.buffer | (1 << (n&0x1f)))
    def pop(self, n):
        return BitArray32(self.buffer & ~(1<<(n&0x1f))), (self.buffer>>(n&0x1f))&1

