import functools
import typing
import jax
import jax.numpy as jnp

INT_MAX = 0x7fffffff

__all__ = 'SortNet',

class SortNet(typing.NamedTuple):
    buffer: jax.Array
    @classmethod
    def init(cls, delay, capacity=None, grad=False):
        return cls(
                jnp.full(delay if capacity is None else capacity, INT_MAX, 'float32' if grad else 'int32'),
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
    buffer = self.buffer
    tmp = []
    for i in range(buffer.shape[0]):
        a, b = buffer[i], buffer[i+1]
        a, b = swap(a, b)
        tmp.append(a)
        tmp.append(b)


    end = self.buffer.shape[0] - 1
    do_insert = self.buffer[end] > n
    return SortNet(
       jax.lax.select(do_insert,
                      jnp.sort(self.buffer.at[end].set(n)),
                      self.buffer)
       )
@_enqueue.defjvp
def _enqueue_jvp(primals, tangents):
    raise NotImplementedError()

@jax.custom_jvp
def _pop(self, n):

@_pop.defjvp
def _pop_jvp(primals, tangents):
    raise NotImplementedError()


def swap(a, b):
    return jax.lax.cond(a < b,
                 lambda: (a, b),
                 lambda: (b, a))


'''
'''
