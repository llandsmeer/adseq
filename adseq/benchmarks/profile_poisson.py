#!/usr/bin/env python
import jax
import json
import numpy as np
import tqdm
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt

NREPEATS = 3

print(__file__)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# jax.config.update('jax_platform_name', 'cpu')

from .. import implementations
from .. import benchmarks

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
    lam = 400 # in units of dt
    delay = 80 # units of dt
    Nevents = 1_00
    stream = mkev(lam, Nevents)
    def f_loop(carry, arg):
        queue, total = carry
        t, ev = arg
        queue, out = queue.pop(t)
        enqueued = queue.enqueue(t + delay)
        queue = jax.tree.map(lambda a, b: jnp.where(ev, a, b), enqueued, queue)
        total = total + out
        return (queue, total), None
    runner = benchmarks.mkrunner_loop(
            f_loop,
            init=(QueueT.init(delay), 0),
            xs=stream,
            groq_unroll=100
            )
    runs = []
    for _ in range(NREPEATS):
        a = time.time()
        o = runner()
        b = time.time()
        if isinstance(o, Exception):
            print(QueueT)
            print(repr(o))
        else:
            runs.append(b-a)
    return np.mean(np.array(runs)) / stream.shape[0] * 1e6

def time_queue_batched(
        QueueT: type[implementations.BaseQueue],
        num=10000,
        nevents=100,
        # assume dt = 0.025
        lam = 400, # 100 Hz
        delay = 80 # 2 ms
        ):
    Nevents = nevents
    key = jax.random.PRNGKey(0)
    @jax.jit
    def f_loop(carry, _):
        del _
        qs, total, key, t = carry
        key, key_next = jax.random.split(key)
        evs = jax.random.uniform(key, shape=(num,)) < 1 / lam
        queue, out = jax.vmap(lambda q: q.pop(t))(qs)
        queue = jax.vmap(lambda e, q: jax.tree.map(
            lambda a, b: jnp.where(e, a, b), q.enqueue(t + delay), q))(evs, qs)
        total = total + out.sum()
        return (queue, total, key_next, t + 1), None
    init = jax.vmap(lambda _: QueueT.init(delay))(jnp.full(num, 0)) # type: ignore
    runner = benchmarks.mkrunner_loop(
            f_loop,
            init=(init, 0, key, 0),
            length=Nevents * lam,
            groq_unroll=20
            )
    runs = []
    for _ in range(NREPEATS):
        a = time.time()
        o = runner()
        b = time.time()
        if isinstance(o, Exception):
            print(repr(o))
        else:
            runs.append(b-a)
    return np.mean(np.array(runs)) / (Nevents * lam) * 1e6

    # implementations.BinaryHeap.sized(7),
    # implementations.LossyRing.sized(4),
    # implementations.FIFORing.sized(4),
    # implementations.SortedArray.sized(4),

check = [
    implementations.BitArray32,
    #implementations.BinaryHeap.sized(2),
    #implementations.BinaryHeap.sized(3),
    implementations.BinaryHeap.sized(7),
    #implementations.BinaryHeap.sized(5),
    #implementations.BinaryHeap.sized(6),
    #implementations.BinaryHeap.sized(7),
    implementations.SingleSpike,
    implementations.SingleSpikeKeep,
    implementations.DoNothing,
    implementations.Ring,
    #implementations.LossyRing.sized(2),
    implementations.LossyRing.sized(4),
    #implementations.LossyRing.sized(100),
    #implementations.FIFORing.sized(2),
    implementations.FIFORing.sized(4),
    #implementations.FIFORing.sized(8),
    #implementations.FIFORing.sized(100),
    #implementations.SortedArray.sized(2),
    implementations.SortedArray.sized(4),
    #implementations.SortedArray.sized(8),
    implementations.BGPQ1,
]

def run():
    results = {
            'single': {},
            'batched': {}
            }
    dev_name, results['host'] = benchmarks.get_device_id()
    print('Single')
    times = []
    finished = []
    for imp in tqdm.tqdm(check):
        print('###', imp.__name__)
        try:
            times.append(time_queue_single(imp))
            finished.append(imp)
        except Exception as ex:
            print(repr(ex))
    assert len(times) == len(finished)
    for t, imp in sorted(zip(times, finished)):
        print(imp.__name__.ljust(20), f'{t: 10.7f}us/ts')
        results['single'][str(imp.__name__)] = float(t)
    print()
    print('Batched')
    times = []
    finished = []
    for imp in (bar := tqdm.tqdm(check)):
        print('###', imp.__name__)
        try:
            t = time_queue_batched(imp)
            print(' (prelim)', imp.__name__.ljust(20), f'{t: 10.7f}us/ts')
            times.append(t)
            finished.append(imp)
        except Exception as ex:
            print(repr(ex))
    assert len(times) == len(finished)
    for t, imp in sorted(zip(times, finished), key=lambda x:x[0]):
        print(imp.__name__.ljust(20), f'{t: 10.7f}us/ts')
        results['batched'][str(imp.__name__)] = float(t)
    with open(f'benchmarks/{dev_name}.json', 'w') as f:
        json.dump(results, f)
    input('done')

def run_increasing_caps():
    times = []
    finished = []
    for imp in (bar := tqdm.tqdm(check)):
        print('###', imp.__name__)
        try:
            t = time_queue_batched(imp)
            print(' (prelim)', imp.__name__.ljust(20), f'{t: 10.7f}us/ts')
            times.append(t)
            finished.append(imp)
        except Exception as ex:
            print(repr(ex))
    assert len(times) == len(finished)
    for t, imp in sorted(zip(times, finished), key=lambda x:x[0]):
        print(imp.__name__.ljust(20), f'{t: 10.7f}us/ts')
        results['batched'][str(imp.__name__)] = float(t)

def run_increasing_sizes():
    dev_name, host = benchmarks.get_device_id()
    with open(f'benchmarks/sizes_{dev_name}.json', 'w') as f:
        for n in range(1000, 1000_000, 1000):
            for imp in (bar := tqdm.tqdm(check)):
                if n > 10_000 and not (n % 10_000) == 0:
                    continue
                if n > 100_000 and not (n % 100_000) == 0:
                    continue
                print('###', imp.__name__)
                try:
                    t = time_queue_batched(imp, num=n, nevents=10)
                    print(n, imp.__name__, t, file=f, flush=True)
                    print(' (prelim)', n, imp.__name__.ljust(20), f'{t: 10.7f}us/ts')
                except Exception as ex:
                    print(repr(ex))

def run_increasing_caps():
    qs = [
        implementations.BinaryHeap,
        implementations.LossyRing,
        implementations.FIFORing,
        implementations.SortedArray,
    ]
    dev_name, host = benchmarks.get_device_id()
    with open(f'benchmarks/caps_{dev_name}.json', 'w') as f:
        for n in range(1, 40):
            for imp in (bar := tqdm.tqdm(qs)):
                iimp = imp.sized(n)
                print('###', iimp.__name__)
                try:
                    t = time_queue_batched(iimp, num=10000, nevents=100, lam=10, delay=5)
                    print(n, iimp.__name__, t, file=f, flush=True)
                    print(' (prelim)', n, iimp.__name__.ljust(20), f'{t: 10.7f}us/ts')
                except Exception as ex:
                    print(repr(ex))

if __name__ == '__main__':
    run_increasing_caps()
    run_increasing_sizes()

