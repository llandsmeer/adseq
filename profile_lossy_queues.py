import jax
import jax.numpy as jnp
import tqdm
import matplotlib.pyplot as plt

import implementations

def mkev(lam: float, Nevents: int, key = jax.random.PRNGKey(0)):
    Ntimesteps = lam * Nevents
    event_stream = jnp.zeros((Ntimesteps, ), dtype=bool)
    ts = jnp.round(jnp.cumulative_sum(
                        jax.random.poisson(key, lam, (Nevents,))
                        )).astype(int)
    return event_stream.at[ts].set(True)

def spike_drop(QueueT, lam=10, delay=1, Nevents=100, drop=20, key=jax.random.PRNGKey(0)):
    stream = mkev(lam, Nevents, key=key)
    @jax.jit
    def f_loop(queue, arg):
        t, ev = arg
        queue, out = queue.pop(t)
        queue = jax.lax.cond(ev, lambda: queue.enqueue(t + delay), lambda: queue)
        return queue, out
    _, trace = jax.lax.scan(f_loop, QueueT.init(delay), xs=(jnp.arange(len(stream)), stream))
    expected = jnp.roll(stream, delay).at[-drop:].set(False)
    # assert trace.max() == 1
    got = trace.astype(bool)
    got = got.at[-drop:].set(False)
    drop = expected.sum() - got.sum()
    # assert (1-expected[got]).sum() == 0 # No FP
    return drop / expected.sum()

def make_plot(imp, show=True, save=False):
    for lam in 10, 20, 50, 200:
        dly = jnp.arange(1, 2*lam)
        o = []
        for i in range(3):
            key=jax.random.PRNGKey(0)
            f = jax.vmap(lambda delay: spike_drop(imp, lam=lam, delay=delay, Nevents=100, drop=20, key=key))
            out = f(dly)
            o.append(out)
        o = jnp.mean(jnp.vstack(o), axis=0)
        plt.plot(dly/lam, o*100, label=f'$\\lambda$={lam}')

    plt.title(imp.__name__)
    plt.legend()
    plt.ylabel('Drop rate (%)')
    plt.xlabel('$d/\\lambda$ or $d\\cdot f$ ')
    plt.ylim(0, 100)
    if save:
        plt.savefig(f'./img/drop_rate_{imp.__name__}.svg')
    if show:
        plt.show()
    plt.clf()



check = [
    implementations.SingleSpike,
    implementations.LossyRing.sized(2),
    implementations.LossyRing.sized(3),
    implementations.LossyRing.sized(4),
    implementations.LossyRing.sized(5),
    implementations.LossyRing.sized(6),
    implementations.LossyRing.sized(7),
    implementations.LossyRing.sized(8),
    implementations.LossyRing.sized(9),
    implementations.LossyRing.sized(10),
    implementations.FIFORing.sized(1),
    implementations.FIFORing.sized(2),
    implementations.FIFORing.sized(3),
    implementations.FIFORing.sized(4),
    implementations.FIFORing.sized(5),
    implementations.FIFORing.sized(6),
    implementations.FIFORing.sized(7),
    implementations.FIFORing.sized(8),
    implementations.FIFORing.sized(9),
    implementations.FIFORing.sized(10),
]

if __name__ == '__main__':
    for imp in tqdm.tqdm(check):
        make_plot(imp, show=False, save=True)

