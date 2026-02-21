from __future__ import annotations
import typing
import jax.numpy as jnp
import jax
from functools import partial

__all__ = 'BGPQ1',

class BGPQ1(typing.NamedTuple):
    inner: SRush_BGPQ_KeyOnly
    @classmethod
    def init(cls, delay):
        del delay
        return BGPQ1(SRush_BGPQ_KeyOnly.make(size=8))
    def enqueue(self, n):
        return BGPQ1(self.inner.insert(8, jnp.array([n]).astype(float)))
    def pop(self, n):
        nxt, ks = self.inner.delete_min(1)
        hit = (ks <= n)[0]
        return (jax.lax.cond(hit, lambda: BGPQ1(nxt), lambda: self),
                hit.astype('int32'))

class SRush_BGPQ_KeyOnly(typing.NamedTuple):
    key_store: jax.Array
    size: jax.Array
    @classmethod
    def make(cls, group=1, size=None):
        assert size is not None
        key_store, size = make_heap(group, size)
        return cls(key_store, size)
    def insert(self, max_size, keys):
        key_store, size = insert(self.key_store, self.size, max_size, keys)
        return SRush_BGPQ_KeyOnly(key_store, size)
    @partial(jax.jit, static_argnums=1)
    def delete_min(self, msize):
        (key_store, size), keys = delete_min(self, msize)
        return SRush_BGPQ_KeyOnly(key_store, size), keys

@partial(jnp.vectorize, signature="(a),(a)->(a),(a)")
def merge(a, b):
    n = a.shape[-1]
    ordera = jnp.searchsorted(a, b) + jnp.arange(n)
    orderb = jnp.searchsorted(b, a, side='right') + jnp.arange(n)
    out = jnp.zeros((a.shape[-1] + b.shape[-1],))
    out = out.at[ordera].set(b)
    out = out.at[orderb].set(a)
    return out[: a.shape[-1]], out[a.shape[-1] :]

# @partial(jax.jit, static_argnums=(1,))
def make_path(index, bits):
    mask = 2**jnp.arange(bits-1, -1, -1)
    x = jnp.bitwise_and(jnp.array(index+1).ravel(), mask) != 0
    def path(c, a):
        x = a + 2 * c
        x = jnp.minimum(x, index+1)
        return x, x
    _, x = jax.lax.scan(path, jnp.array([1]), x[1:])
    return jnp.concatenate((jnp.array([0]), (x - 1).reshape(-1)))

def make_heap(group_size, total_size):
    size = jnp.zeros(1, dtype=int)
    key_store = jnp.full((total_size, group_size), 1.e5)
    return (key_store, size)

# @partial(jax.jit, static_argnums=3)
def insert(key_store, size, max_size, keys):
    path = make_path(size, max_size)
    def insert_heapify(state, n):
        key_store, keys = state
        head, keys = merge(key_store[n], keys)
        return (key_store.at[n].set(head), keys), None
    (key_store, keys), _ = \
        jax.lax.scan(insert_heapify, (key_store, keys), path)
    return key_store, size + 1

# @partial(jax.jit, static_argnums=1)
def delete_min(heap, msize):
    INF = 1.e9
    key_store, size = heap
    keys = key_store[0]
    def one():
        return key_store.at[0].set(INF)
    def two():
        path = make_path(size - 1, msize)
        key_store2 = key_store.at[0].set(key_store[path[-1]]).at[path[-1]].set(INF)
        key_store3, n = \
            jax.lax.fori_loop(0, msize, delete_heapify, (key_store2, 0))
        return key_store3
    key_store = jax.lax.cond((size == 1).all(), one, two)
    size = size - 1
    return (key_store, size), keys

def delete_heapify(_, state):
    key_store, n = state
    c = jnp.stack(((n + 1) * 2 - 1, (n + 1) * 2))
    c_l,c_r = key_store[c[0]], key_store[c[1]]
    ins = jnp.where(c_l[-1] < c_r[-1], 0, 1)
    s, l = c[ins], c[1 - ins]
    small, k2 = merge(c_l, c_r)
    k1, k2 = merge(key_store[n], small)
    key_store = key_store.at[l].set(k2).at[n].set(k1).at[s].set(k2)
    return key_store, s
