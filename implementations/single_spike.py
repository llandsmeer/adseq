import typing
import jax

INT_MAX = 0x7fffffff

__all__ = 'SingleSpike',

class SingleSpike(typing.NamedTuple):
    last_spike: int
    @classmethod
    def init(cls, delay):
        del delay
        return cls(INT_MAX)
    def enqueue(self, n):
        return SingleSpike(n)
    def pop(self, n):
        hit = self.last_spike <= n
        return (jax.lax.cond(hit, lambda: SingleSpike(INT_MAX), lambda: self),
                hit.astype('int32'))
