import jax
import tqdm
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
    # print(QueueT.__name__.ljust(20), f'{b - a: 10.7f}s')
    return b - a

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

times = [time_queue(imp) for imp in tqdm.tqdm(check)]

for t, imp in sorted(zip(times, check)):
    print(imp.__name__.ljust(20), f'{t: 10.7f}s')

input('done')
