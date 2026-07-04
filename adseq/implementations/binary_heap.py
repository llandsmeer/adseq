'''
Assumes sorted inputs, ie homogeneous delays
'''

import functools
import math
import typing
import jax
import jax.numpy as jnp

INT_MAX = 0x7fffffff

__all__ = 'BinaryHeap',

def parent(i): return (i - 1) // 2
def left_child(i): return 2 * i + 1
def right_child(i): return 2 * i + 2

class BinaryHeap(typing.NamedTuple):
    buffer: jax.Array
    size: int | jax.Array
    @classmethod
    def init(cls, delay, capacity=None, grad=False):
        return cls(
                jnp.full(
                    delay if capacity is None else capacity,
                    INT_MAX,
                    'float32'
                    ),
                0,
                )
    @classmethod
    def sized(cls, n):
        'wish I could use __class_getitem__'
        return type(f'{cls.__name__}[{n}]',
                    cls.__bases__,
                    {**cls.__dict__,
                     'init': functools.partial(cls.init, capacity=n)})
    def enqueue(self, n):
        return _enqueue(self, n)
    def pop(self, n):
        return _pop(self, n)

@jax.custom_jvp
def _enqueue(self, n):
    buffer = self.buffer.at[self.size].set(n)
    size = self.size + 1
    buffer = _up(buffer, size-1)
    full = self.size == self.buffer.shape[0]
    return BinaryHeap(
        jnp.where(full, self.buffer, buffer),
        jnp.where(full, self.size, size))

@_enqueue.defjvp
def _enqueue_jvp(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    buffer = self.buffer.at[self.size].set(n)
    buffer_t = self_t.buffer.at[self.size].set(n_t)
    size = self.size + 1
    buffer, buffer_t = _up_dual(buffer, buffer_t, size-1)
    full = self.size == self.buffer.shape[0]
    primal_out = BinaryHeap(
        jnp.where(full, self.buffer, buffer),
        jnp.where(full, self.size, size))
    tangent_out = BinaryHeap(
        jnp.where(full, self_t.buffer, buffer_t),
        self_t.size)
    return primal_out, tangent_out

@jax.custom_jvp
def _pop(self, n):
    root = self.buffer[0]
    hit = (self.size > 0) & (n >= root)
    buffer = self.buffer.at[0].set(self.buffer[self.size-1])
    size = self.size - 1
    buffer = _down(size, buffer, 0)
    inner_buffer = jnp.where(self.size == 1, self.buffer, buffer)
    return (BinaryHeap(
                jnp.where(hit, inner_buffer, self.buffer),
                jnp.where(hit, self.size - 1, self.size)),
            hit.astype(self.buffer.dtype))

@_pop.defjvp
def _pop_jvp(primals, tangents):
    self, n = primals
    self_t, n_t = tangents
    del n_t
    root = self.buffer[0]
    hit = (self.size > 0) & (n >= root)
    buffer = self.buffer.at[0].set(self.buffer[self.size-1])
    buffer_t = self_t.buffer.at[0].set(self_t.buffer[self.size-1])
    size = self.size - 1
    buffer, buffer_t = _down_dual(size, buffer, buffer_t, 0)
    inner_buffer_p = jnp.where(self.size == 1, self.buffer, buffer)
    inner_buffer_t = jnp.where(self.size == 1, self_t.buffer, buffer_t)
    primal_out = (BinaryHeap(
                      jnp.where(hit, inner_buffer_p, self.buffer),
                      jnp.where(hit, self.size - 1, self.size)),
                  hit.astype(self.buffer.dtype))
    tangent_out = (BinaryHeap(
                   jnp.where(hit, inner_buffer_t, self_t.buffer),
                   self_t.size),
               jnp.where(hit, self_t.buffer[0], jnp.array(0., dtype=self.buffer.dtype)))
    return primal_out, tangent_out


del _enqueue_jvp, _pop_jvp

def _swap(buffer, current, smallest):
    a, b = buffer[current], buffer[smallest]
    return buffer.at[current].set(a).at[smallest].set(b)

def _down(size, buffer, i):
    current = i
    l, r = left_child(current), right_child(current)
    smallest = current
    smallest = jax.lax.select((l < size) & (buffer[l] < buffer[smallest]), l, smallest)
    smallest = jax.lax.select((r < size) & (buffer[r] < buffer[smallest]), r, smallest)
    def cond(x):
        smallest, current, buffer = x
        del buffer
        return smallest != current
    def body(x):
        smallest, current, buffer = x
        buffer = _swap(buffer, current, smallest)
        current = smallest
        l, r = left_child(current), right_child(current)
        smallest = current
        smallest = jax.lax.select((l < size) & (buffer[l] < buffer[smallest]), l, smallest)
        smallest = jax.lax.select((r < size) & (buffer[r] < buffer[smallest]), r, smallest)
        return smallest, current, buffer
    return jax.lax.while_loop(cond, body, (smallest, current, buffer))[2]

def _up(buffer, i):
    def cond(x):
        i, buffer = x
        return (i != 0) & (buffer[parent(i)] > buffer[i])
    def body(x):
        i, buffer = x
        buffer = _swap(buffer, i, parent(i))
        i = parent(i)
        return i, buffer
    return jax.lax.while_loop(cond, body, (i, buffer))[1]

def _down_dual(size, buffer, buffer_t, i):
    current = i
    l, r = left_child(current), right_child(current)
    smallest = current
    smallest = jax.lax.select((l < size) & (buffer[l] < buffer[smallest]), l, smallest)
    smallest = jax.lax.select((r < size) & (buffer[r] < buffer[smallest]), r, smallest)
    if isinstance(buffer_t, jax.core.Tracer):
        cap = buffer.shape[0]
        max_iters = int(math.log2(max(cap, 2))) + 1
        def scan_body(carry, _):
            smallest, current, buffer, buffer_t, done = carry
            should_swap = (~done) & (smallest != current)
            swapped_buffer = _swap(buffer, current, smallest)
            swapped_buffer_t = _swap(buffer_t, current, smallest)
            new_current = jnp.where(should_swap, smallest, current)
            new_buffer = jnp.where(should_swap, swapped_buffer, buffer)
            new_buffer_t = jnp.where(should_swap, swapped_buffer_t, buffer_t)
            l, r = left_child(new_current), right_child(new_current)
            new_smallest = new_current
            new_smallest = jax.lax.select((l < size) & (new_buffer[l] < new_buffer[new_smallest]), l, new_smallest)
            new_smallest = jax.lax.select((r < size) & (new_buffer[r] < new_buffer[new_smallest]), r, new_smallest)
            new_done = done | (~should_swap)
            return (new_smallest, new_current, new_buffer, new_buffer_t, new_done), None
        init = (smallest, current, buffer, buffer_t, jnp.array(False))
        (_, _, buffer, buffer_t, _), _ = jax.lax.scan(scan_body, init, None, length=max_iters)
    else:
        def cond(x):
            smallest, current, buffer, buffer_t = x
            del buffer_t, buffer
            return smallest != current
        def body(x):
            smallest, current, buffer, buffer_t = x
            buffer = _swap(buffer, current, smallest)
            buffer_t = _swap(buffer_t, current, smallest)
            current = smallest
            l, r = left_child(current), right_child(current)
            smallest = current
            smallest = jax.lax.select((l < size) & (buffer[l] < buffer[smallest]), l, smallest)
            smallest = jax.lax.select((r < size) & (buffer[r] < buffer[smallest]), r, smallest)
            return smallest, current, buffer, buffer_t
        _, _, buffer, buffer_t = jax.lax.while_loop(cond, body, (smallest, current, buffer, buffer_t))
    return buffer, buffer_t

def _up_dual(buffer, buffer_t, i):
    if isinstance(buffer_t, jax.core.Tracer):
        cap = buffer.shape[0]
        max_iters = int(math.log2(max(cap, 2))) + 1
        def scan_body(carry, _):
            i, buffer, buffer_t, done = carry
            pi = parent(i)
            should_swap = (~done) & (i != 0) & (buffer[pi] > buffer[i])
            swapped_buffer = _swap(buffer, i, pi)
            swapped_buffer_t = _swap(buffer_t, i, pi)
            new_buffer = jnp.where(should_swap, swapped_buffer, buffer)
            new_buffer_t = jnp.where(should_swap, swapped_buffer_t, buffer_t)
            new_i = jnp.where(should_swap, pi, i)
            new_done = done | (~should_swap)
            return (new_i, new_buffer, new_buffer_t, new_done), None
        init = (i, buffer, buffer_t, jnp.array(False))
        (_, out, out_t, _), _ = jax.lax.scan(scan_body, init, None, length=max_iters)
    else:
        def cond(x):
            i, buffer, buffer_t = x
            return (i != 0) & (buffer[parent(i)] > buffer[i])
        def body(x):
            i, buffer, buffer_t = x
            i_up = parent(i)
            buffer = _swap(buffer, i, i_up)
            buffer_t = _swap(buffer_t, i, i_up)
            i = parent(i)
            return i, buffer, buffer_t
        _, out, out_t = jax.lax.while_loop(cond, body, (i, buffer, buffer_t))
    return out, out_t



# @jax.custom_jvp
# def _enqueue(self, n):
#     cap = self.buffer.shape[0]
#     do_insert = self.size < cap
#     return BinaryHeap(
#        jax.lax.select(do_insert, self.buffer.at[(self.head + self.size) % cap].set(n), self.buffer),
#        self.head,
#        jax.lax.select(do_insert, self.size+1, self.size)
#        )
# @_enqueue.defjvp
# def _enqueue_jvp(primals, tangents):
#     self, n = primals
#     self_t, n_t = tangents
#     cap = self.buffer.shape[0]
#     do_insert = self.size < cap
#     return BinaryHeap(
#                jax.lax.select(do_insert, self.buffer.at[(self.head + self.size) % cap].set(n), self.buffer),
#                self.head,
#                jax.lax.select(do_insert, self.size+1, self.size)
#        ),  BinaryHeap(
#                jax.lax.select(do_insert, self_t.buffer.at[(self.head + self.size) % cap].set(n_t), self_t.buffer),
#                self_t.head,
#                self_t.size
#        )
# del _enqueue_jvp
# 
# @jax.custom_jvp
# def _pop(self, n):
#     cap = self.buffer.shape[0]
#     hit = self.buffer[self.head] <= n
#     return BinaryHeap(
#        jax.lax.select(hit, self.buffer.at[self.head].set(INT_MAX), self.buffer),
#        jax.lax.select(hit, (self.head+1) % cap, self.head),
#        jax.lax.select(hit, self.size-1, self.size)
#        ), hit.astype(self.buffer.dtype)
# 
# @_pop.defjvp
# def _pop_jvp(primals, tangents):
#     self, n = primals
#     self_t, n_t = tangents
#     del n_t
#     cap = self.buffer.shape[0]
#     hit = self.buffer[self.head] <= n
#     return (BinaryHeap(
#                jax.lax.select(hit, self.buffer.at[self.head].set(INT_MAX), self.buffer),
#                jax.lax.select(hit, (self.head+1) % cap, self.head),
#                jax.lax.select(hit, self.size-1, self.size)
#        ), hit.astype(self.buffer.dtype)), (
#            BinaryHeap(
#                jax.lax.select(hit, self_t.buffer.at[self.head].set(INT_MAX), self_t.buffer),
#                self_t.head,
#                self_t.size
#        ), self_t.buffer[self.head])
# del _pop_jvp
