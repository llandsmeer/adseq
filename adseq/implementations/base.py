import jax
from typing import Tuple, Protocol, runtime_checkable
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

@runtime_checkable
class BaseQueue(Protocol):
    @classmethod
    def init(cls, delay: int) -> Self:
        ...

    def enqueue(self, n: int) -> Self:
        ...

    def pop(self, n: int) -> Tuple[Self, int | jax.Array]:
        ...
