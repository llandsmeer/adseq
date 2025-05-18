import inspect
import pytest

import jax
import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import implementations

check = [
    implementations.BinaryHeap,
    implementations.BGPQ1,
    implementations.SingleSpike,
    implementations.SingleSpikeKeep,
    implementations.Ring,
    implementations.LossyRing.sized(2),
    implementations.LossyRing.sized(4),
    implementations.LossyRing.sized(100),
    implementations.FIFORing.sized(2),
    implementations.FIFORing.sized(4),
    implementations.FIFORing.sized(8),
    implementations.FIFORing.sized(100),
    implementations.SortedArray.sized(2),
    implementations.SortedArray.sized(5),
    implementations.BitArray32
]

key = jax.random.PRNGKey(0)

def mkev(lam: float, Nevents: int):
    Ntimesteps = lam * Nevents
    event_stream = jnp.zeros((Ntimesteps, ), dtype=bool)
    ts = jnp.round(jnp.cumulative_sum(
                        jax.random.poisson(key, lam, (Nevents,))
                        )).astype(int)
    return event_stream.at[ts].set(True)

named_check = {imp.__name__:imp for imp in check}

def generic_test_queue(QueueT):
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
    return ok.mean()

@pytest.mark.parametrize("imp_name", named_check)
def test_queue(imp_name):
    assert generic_test_queue(named_check[imp_name]) > 0.95

def test_do_nothing():
    assert generic_test_queue(implementations.DoNothing) < 0.95

def test_coverage():
    checked = [nm.split('[')[0] for nm in named_check.keys()]
    for k, cls in vars(implementations).items():
        if k.startswith('__'):
            continue
        if cls is implementations.DoNothing:
            continue
        if cls is implementations.BaseQueue:
            continue
        if isinstance(cls, type) and \
           issubclass(cls, implementations.BaseQueue):
            assert k in checked
