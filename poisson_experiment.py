import jax
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt

import implementations

key = jax.random.PRNGKey(0)

dtype_ev = 'int32'

# timestamps are encoded as 32-bit timesteps in units of dt

def mkev(lam: float, Nevents: int):
    Ntimesteps = lam * Nevents
    event_stream = jnp.zeros((Ntimesteps, ), dtype=bool)
    ts = jnp.round(jnp.cumulative_sum(
                        jax.random.poisson(key, lam, (Nevents,))
                        )).astype(dtype_ev)
    return event_stream.at[ts].set(True)


def test_queue(QueueT: type[implementations.BaseQueue]):
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
    _, trace = jax.lax.scan(f_loop, QueueT.init(delay), xs=(jnp.arange(len(stream)), stream))
    ok = (jnp.roll(stream, delay) == trace)[:-delay]
    if ok.mean() < 0.99:
        print('not good', QueueT.__name__)
    try:
        assert ok.mean() > 0.90
    except AssertionError:
        print(":(")

def time_queue(QueueT: type[implementations.BaseQueue]):
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

    f = lambda stream:jax.lax.scan(
            f=f_loop,
            init=(QueueT.init(delay), 0),
            xs=(jnp.arange(len(stream)), stream)
            )[0][1]
    f = jax.jit(f)
    f(stream)
    a = time.time()
    f(stream)
    b = time.time()
    print(QueueT.__name__.ljust(20), f'{b - a: 10.7f}s')

test_queue(implementations.BGPQ1)
test_queue(implementations.SingleSpike)
test_queue(implementations.DoNothing)
test_queue(implementations.Ring)
test_queue(implementations.LossyRing.sized(8))

time_queue(implementations.BGPQ1)
time_queue(implementations.SingleSpike)
time_queue(implementations.DoNothing)
time_queue(implementations.Ring)
time_queue(implementations.LossyRing.sized(2))
time_queue(implementations.LossyRing.sized(4))
time_queue(implementations.LossyRing.sized(100))

input('done')
