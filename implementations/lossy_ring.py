import functools
import typing
from .ring import Ring

class LossyRing(typing.NamedTuple):
    ring: Ring
    @classmethod
    def init(cls, delay, wrap=4):
        del delay
        return LossyRing(Ring.init(delay=wrap))
    def enqueue(self, n):
        return LossyRing(self.ring.enqueue(n))
    def pop(self, n):
        q, n = self.ring.pop(n)
        return LossyRing(q), n
    @classmethod
    def sized(cls, n):
        "wish I could use __class_getitem__"
        return type(f'{cls.__name__}[{n}]',
                    cls.__bases__,
                    {**cls.__dict__,
                     "init": functools.partial(cls.init, wrap=n)})
