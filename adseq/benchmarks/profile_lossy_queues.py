#!/usr/bin/env python

import jax
import jax.numpy as jnp
import tqdm
import matplotlib.pyplot as plt

import sys, os

from .. import implementations

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

def make_plot(imp, ax):
    for lam in 10, 20, 50, 200:
        dly = jnp.arange(1, 2*lam)
        o = []
        for i in range(3):
            key=jax.random.PRNGKey(0)
            f = jax.vmap(lambda delay: spike_drop(imp, lam=lam, delay=delay, Nevents=100, drop=20, key=key))
            out = f(dly)
            o.append(out)
        o = jnp.mean(jnp.vstack(o), axis=0)
        ax.plot(dly/lam, o*100, label=f'$\\lambda$={lam}')
    ax.set_title(imp.__name__)
    # ax.legend()
    ax.set_ylabel('Drop rate (%)')
    ax.set_xlabel('$d/\\lambda$ or $d\\cdot f$ ')
    ax.set_ylim(0, 100)



check = [
    implementations.SingleSpike,
    implementations.SingleSpikeKeep,
    implementations.FIFORing.sized(2),
    implementations.FIFORing.sized(3),
    implementations.LossyRing.sized(2),
    implementations.LossyRing.sized(3),
    implementations.LossyRing.sized(4),
    implementations.LossyRing.sized(5),
]

if __name__ == '__main__':
    n = len(check)
    nc = 8
    nr = int(jnp.ceil(n / nc))
    fig, ax= plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True,
                          gridspec_kw=dict(hspace=.22, wspace=0), squeeze=False)
    aax = ax.flatten()
    for i, imp in enumerate(tqdm.tqdm(check)):
        make_plot(imp, ax=aax[i])

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
    #plt.savefig(f'./img/lossy_queues.svg')
    #plt.savefig(f'./img/lossy_queues.png')
    plt.show()

