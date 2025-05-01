import jax
import tqdm
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt

import implementations

# timestamps are encoded as 32-bit timesteps in units of dt

def mkev(lam: float, Nevents: int, key=jax.random.PRNGKey(0)):
    Ntimesteps = lam * Nevents
    event_stream = jnp.zeros((Ntimesteps, ), dtype=bool)
    ts = jnp.round(jnp.cumulative_sum(
                        jax.random.poisson(key, lam, (Nevents,))
                        )).astype(int)
    return event_stream.at[ts].set(True)

def mkevs(lam, Nevents, num, key=jax.random.PRNGKey(0)):
    keys = jax.random.split(key, num)
    return jax.vmap(lambda k: mkev(lam, Nevents, k))(keys)

def time_queue_single(QueueT: type[implementations.BaseQueue]):
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
    f(stream)
    f(stream)
    a = time.time()
    f(stream)
    b = time.time()
    # print(QueueT.__name__.ljust(20), f'{b - a: 10.7f}s')
    return (b - a) / stream.shape[0] * 1e6

def time_queue_batched(QueueT: type[implementations.BaseQueue]):
    # assume dt = 0.025
    stupid_sum = 0
    lam = 400 # 100 Hz
    delay = 80 # 2 ms
    Nevents = 100
    num = 1000
    stream = mkevs(lam, Nevents, num)
    @jax.jit
    def f_loop(carry, arg):
        qs, total = carry
        t, evs = arg
        queue, out = jax.vmap(lambda q: q.pop(t))(qs)
        queue = jax.vmap(lambda e, q: jax.lax.cond(e, lambda: q.enqueue(t + delay), lambda: q))(evs, qs)
        total = total + out.sum()
        return (queue, total), None
    init = jax.vmap(lambda _: QueueT.init(delay))(jnp.full(num, 0)) # type: ignore
    f = lambda stream:jax.lax.scan(
            f=f_loop,
            init=(init, 0),
            xs=(jnp.arange(stream.shape[1]), stream.T)
            )[0][1]
    f = jax.jit(f)
    stupid_sum += f(stream)
    a = time.time()
    stream = mkevs(lam, Nevents, num, key=jax.random.PRNGKey(1))
    stupid_sum += f(stream)
    b = time.time()
    return (b - a) / stream.shape[1] * 1e6

check = [
    implementations.BGPQ1,
    implementations.SingleSpike,
    implementations.DoNothing,
    implementations.Ring,
    implementations.LossyRing.sized(2),
    implementations.LossyRing.sized(4),
    implementations.LossyRing.sized(100),
    implementations.FIFORing.sized(2),
    implementations.FIFORing.sized(4),
    implementations.FIFORing.sized(8),
    implementations.FIFORing.sized(100),
]

print('Single')
times = [time_queue_single(imp) for imp in tqdm.tqdm(check)]
for t, imp in sorted(zip(times, check)):
    print(imp.__name__.ljust(20), f'{t: 10.7f}us/ts')
print()
print('Batched')
times = [time_queue_batched(imp) for imp in tqdm.tqdm(check)]
for t, imp in sorted(zip(times, check), key=lambda x:x[0]):
    print(imp.__name__.ljust(20), f'{t: 10.7f}us/ts')

input('done')
