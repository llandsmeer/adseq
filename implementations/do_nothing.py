import typing

__all__ = 'DoNothing',

class DoNothing(typing.NamedTuple):
    @classmethod
    def init(cls, delay):
        del delay
        return cls()
    def enqueue(self, n):
        del n
        return DoNothing()
    def pop(self, n):
        del n
        return DoNothing(), 0
