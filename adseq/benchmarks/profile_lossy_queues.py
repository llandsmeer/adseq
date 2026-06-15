#!/usr/bin/env python
import pandas as pd

import jax
import jax.numpy as jnp
import tqdm
import matplotlib.pyplot as plt

import sys, os

from adseq import implementations

def mkev(lam: float, Nevents: int, key = jax.random.PRNGKey(0)):
    T = int(1/lam * Nevents)
    counts = jax.random.poisson(key, lam=lam, shape=(T,))
    dropped = ((counts > 1) * counts).sum()
    jax.debug.print('generator lam={} dropped={} N={}', lam, dropped, Nevents)
    return (counts > 0).astype(jnp.int32)

def spike_drop(QueueT, lam=10, delay=1, Nevents=100, drop=20, key=jax.random.PRNGKey(0)):
    stream = mkev(lam, Nevents, key=key)
    @jax.jit
    def f_loop(queue, arg):
        t, ev = arg
        queue, out = queue.pop(t)
        queue = jax.lax.cond(ev, lambda: queue.enqueue(t + delay), lambda: queue)
        return queue, out
    _, trace = jax.lax.scan(f_loop, QueueT.init(delay), xs=(jnp.arange(len(stream)), stream))
    expected = jnp.roll(stream.at[-drop:].set(False), delay).at[-drop:].set(False)
    # assert trace.max() == 1
    got = trace.astype(bool)
    got = got.at[-drop:].set(False)
    drop = expected.sum() - got.sum()
    # assert (1-expected[got]).sum() == 0 # No FP
    return drop / expected.sum()

def make_plot(imp, ax, outlist):
    for lam in [0.1, 0.05]:
        dly = jnp.arange(1, int(round(10/lam)))
        o = []
        key = jax.random.PRNGKey(0)
        f = jax.jit(lambda delay, key: jax.vmap(lambda delay: spike_drop(imp, lam=lam, delay=delay, Nevents=1000, drop=200, key=key))(delay))
        assert dly.max() <= 200
        for i in range(3):
            key, k = jax.random.split(key)
            out = f(dly, k)
            o.append(out)
        o = jnp.mean(jnp.vstack(o), axis=0)
        for a, b in zip(dly, o):
            outlist.append(dict(queue=imp.__name__, delay=float(a), lam=float(lam), drop_rate=float(b)))
        ax.plot(dly*lam, o*100, label=f'$\\lambda$={lam}')
    ax.set_title(imp.__name__)
    # ax.legend()
    ax.set_ylabel('Drop rate (%)')
    ax.set_xlabel('$d \\lambda$')
    ax.set_ylim(0, 100)



check = [
    implementations.SingleSpike,
    implementations.SingleSpikeKeep,
    # implementations.FIFORing.sized(2),
    # implementations.FIFORing.sized(3),
    # implementations.LossyRing.sized(2),
    # implementations.LossyRing.sized(3),
    # implementations.LossyRing.sized(4),
    # implementations.LossyRing.sized(5),
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
    implementations.FIFORing.sized(11),
    implementations.FIFORing.sized(12),
    implementations.FIFORing.sized(13),
    implementations.FIFORing.sized(14),
    implementations.FIFORing.sized(15),
    implementations.FIFORing.sized(16),
    implementations.FIFORing.sized(17),
    implementations.FIFORing.sized(18),
    implementations.FIFORing.sized(19),
    implementations.FIFORing.sized(20),
    implementations.FIFORing.sized(21),
    implementations.FIFORing.sized(22),
    implementations.FIFORing.sized(23),
    implementations.FIFORing.sized(24),
    implementations.FIFORing.sized(25),
    implementations.FIFORing.sized(26),
    implementations.FIFORing.sized(27),
    implementations.FIFORing.sized(28),
    implementations.FIFORing.sized(29),
    implementations.FIFORing.sized(30),
    implementations.FIFORing.sized(31),
    implementations.FIFORing.sized(32),
    implementations.FIFORing.sized(33),
    implementations.FIFORing.sized(34),
    implementations.FIFORing.sized(35),
    implementations.FIFORing.sized(36),
    implementations.FIFORing.sized(37),
    implementations.FIFORing.sized(38),
    implementations.FIFORing.sized(39),
    implementations.FIFORing.sized(40),
    implementations.FIFORing.sized(41),
    implementations.FIFORing.sized(42),
    implementations.FIFORing.sized(43),
    implementations.FIFORing.sized(44),
    implementations.FIFORing.sized(45),
    implementations.FIFORing.sized(46),
    implementations.FIFORing.sized(47),
    implementations.FIFORing.sized(48),
    implementations.FIFORing.sized(49),
    implementations.FIFORing.sized(50),
    #
    implementations.LossyRing.sized(1),
    implementations.LossyRing.sized(2),
    implementations.LossyRing.sized(3),
    implementations.LossyRing.sized(4),
    implementations.LossyRing.sized(5),
    implementations.LossyRing.sized(6),
    implementations.LossyRing.sized(7),
    implementations.LossyRing.sized(8),
    implementations.LossyRing.sized(9),
    implementations.LossyRing.sized(10),
    implementations.LossyRing.sized(11),
    implementations.LossyRing.sized(12),
    implementations.LossyRing.sized(13),
    implementations.LossyRing.sized(14),
    implementations.LossyRing.sized(15),
    implementations.LossyRing.sized(16),
    implementations.LossyRing.sized(17),
    implementations.LossyRing.sized(18),
    implementations.LossyRing.sized(19),
    implementations.LossyRing.sized(20),
    implementations.LossyRing.sized(21),
    implementations.LossyRing.sized(22),
    implementations.LossyRing.sized(23),
    implementations.LossyRing.sized(24),
    implementations.LossyRing.sized(25),
    implementations.LossyRing.sized(26),
    implementations.LossyRing.sized(27),
    implementations.LossyRing.sized(28),
    implementations.LossyRing.sized(29),
    implementations.LossyRing.sized(30),
    implementations.LossyRing.sized(31),
    implementations.LossyRing.sized(32),
    implementations.LossyRing.sized(33),
    implementations.LossyRing.sized(34),
    implementations.LossyRing.sized(35),
    implementations.LossyRing.sized(36),
    implementations.LossyRing.sized(37),
    implementations.LossyRing.sized(38),
    implementations.LossyRing.sized(39),
    implementations.LossyRing.sized(40),
    implementations.LossyRing.sized(41),
    implementations.LossyRing.sized(42),
    implementations.LossyRing.sized(43),
    implementations.LossyRing.sized(44),
    implementations.LossyRing.sized(45),
    implementations.LossyRing.sized(46),
    implementations.LossyRing.sized(47),
    implementations.LossyRing.sized(48),
    implementations.LossyRing.sized(49),
    implementations.LossyRing.sized(50),
]

if __name__ == '__main__':
    n = len(check)
    nc = 8
    out = []
    nr = int(jnp.ceil(n / nc))
    fig, ax= plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True,
                          gridspec_kw=dict(hspace=.22, wspace=0), squeeze=False)
    aax = ax.flatten()
    for i, imp in enumerate(tqdm.tqdm(check)):
        make_plot(imp, ax=aax[i], outlist=out)
    df = pd.DataFrame(out)
    df.to_feather('lossy_queues.feather')
    df.to_csv('lossy_queues.csv')

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            if i != ax.shape[0]-1:
                ax[i,j].set_xlabel('')
            if j != 0:
                ax[i,j].set_ylabel('')
    handles, labels = aax[0].get_legend_handles_labels()
    plt.rcParams['pdf.fonttype'] = 42
    fig.legend(handles, labels, loc='upper center', ncol=4)
    plt.tight_layout()
    plt.savefig(f'lossy_queues_v2.svg')
    plt.savefig(f'lossy_queues_v2.png')
    plt.show()

