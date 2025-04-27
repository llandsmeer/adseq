import jax
import time
import typing
import jax.numpy as jnp
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(0)

dtype_ev = 'int32'
INT_MAX = 0x7fffffff

# timestamps are encoded as 32-bit timesteps in units of dt

def mkev(lam, Nevents):
    Ntimesteps = lam * Nevents
    event_stream = jnp.zeros((Ntimesteps, ), dtype=bool)
    ts = jnp.round(jnp.cumulative_sum(
                        jax.random.poisson(key, lam, (Nevents,))
                        )).astype(dtype_ev)
    return event_stream.at[ts].set(True)

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

class Ring(typing.NamedTuple):
    buffer: jax.Array
    @classmethod
    def init(cls, delay):
        return cls(jnp.full(delay, 0, 'int32'))
    def enqueue(self, n):
        delay = self.buffer.shape[0]
        return Ring(self.buffer.at[(n+1) % delay].add(1))
    def pop(self, n):
        delay = self.buffer.shape[0]
        return Ring(self.buffer.at[n % delay].set(0)), \
                self.buffer[n % delay]

def test_queue(QueueT):
    lam = 10 # in units of dt
    delay = 1 # units of dt
    Nevents = 100
    stream = mkev(lam, Nevents)
    @jax.jit
    def f_loop(queue, arg):
        t, ev = arg
        queue, out = queue.pop(t)
        queue = jax.lax.cond(ev, lambda: queue.enqueue(t + delay), lambda: queue)
        return queue, out
    _, trace = jax.lax.scan(f_loop, QueueT.init(), xs=(jnp.arange(len(stream)), stream))
    ok = (jnp.roll(stream, delay) == trace)[:-delay]
    assert ok.mean() > 0.99

def time_queue(QueueT):
    lam = 1000 # in units of dt
    delay = 100 # units of dt
    Nevents = 1_000
    stream = mkev(lam, Nevents)
    @jax.jit
    def f_loop(carry, arg):
        queue, total = carry
        t, ev = arg
        queue, out = queue.pop(t)
        queue = jax.lax.cond(ev, lambda: queue.enqueue(t + delay), lambda: queue)
        total = total + out
        return (queue, total), None

    f = lambda stream:jax.lax.scan(f_loop, (QueueT.init(delay), 0), xs=(jnp.arange(len(stream)), stream))[0][1]
    f = jax.jit(f)

    f(stream)

    a = time.time()
    f(stream)
    b = time.time()
    print(b - a)


time_queue(SingleSpike)
time_queue(DoNothing)
time_queue(Ring)
input()
