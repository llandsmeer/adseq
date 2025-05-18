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
        return LossyRing(self.buffer.at[n % cap].set(n, mode='promise_in_bounds'))
    def pop(self, n):
        # this one broken with the dymnamic slice
        cap = self.buffer.shape[0]
        # root = jax.lax.gather(
        #     self.buffer,
        #     jnp.array([n%cap]),
        #     jax.lax.GatherDimensionNumbers(
        #         offset_dims=(),
        #         collapsed_slice_dims=(0,),
        #         start_index_map=(0,)),
        #     slice_sizes=(1,),
        #     mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS)
        root = self.buffer.at[n%cap].get(mode='promise_in_bounds')
        # root = self.buffer[n%cap]
        hit = root <= n # HERE ONNX CREATES DYNAMIC SLICE
        return LossyRing(
                jax.lax.select(hit,
                    # jax.lax.scatter(
                    #     self.buffer,
                    #     jnp.array([n % cap]),
                    #     INT_MAX,
                    #     jax.lax.ScatterDimensionNumbers((), (0,), (0,), (), ())),
                    self.buffer.at[n % cap].set(INT_MAX, mode='promise_in_bounds'),
                    self.buffer)), hit.astype(int)
    @classmethod
    def sized(cls, n):
        "wish I could use __class_getitem__"
        return type(f'{cls.__name__}[{n}]',
                    cls.__bases__,
                    {**cls.__dict__,
                     "init": functools.partial(cls.init, capacity=n)})
