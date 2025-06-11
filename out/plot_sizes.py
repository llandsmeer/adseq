import pandas as pd
import matplotlib.pyplot as plt

fns = dict(
    tpu='./sizes_t1v-n-0a7ba72c-w-0_tpu_jax.json',
    gpu='./sizes_gcn135.local.snellius.surf.nl_gpu_jax.json',
    cpu='./sizes_spectre_cpu_jax.json'
    )

dfs = {key:
    pd.read_csv(fn, sep=' ', header=None, names=['n', 'q', 't'])
               .set_index(['n', 'q'])
    for key, fn in fns.items()}

df = pd.concat(dfs, names='a')

labels = dict(
        tpu='TPUv4',
        gpu='H100',
        cpu='i7-1195G7'
        )

qs = df.index.get_level_values('q').unique()

order = [
       'Ring',
       'BitArray32',
       'SingleSpike',
       'SingleSpikeKeep',
       'BinaryHeap[7]',
       'LossyRing[4]',
       'FIFORing[4]',
       'SortedArray[4]',
       'BGPQ1',
       ]

fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, gridspec_kw=dict(
    hspace=0.0,wspace=0), figsize=(8, 4))
figlines = []
figlabels = []
for i, (k, sub) in enumerate(df.groupby('q')):
    for j, (a, ssub) in enumerate(sub.groupby('a')):
        if k in 'DoNothing':
            continue
        print(k)
        if k in 'SingleSpike SingleSpikeKeep BitArray32':
            aa = ax[0][j]
            aa.set_title(a)
        elif k in 'BinaryHeap SortedArray[4] BinaryHeap[7]':
            aa = ax[2][j]
        else:
            aa = ax[1][j]
        n = ssub.index.get_level_values('n')
        t = ssub.t
        line, = aa.plot(n, t, label=k)
        if j == 1:
            figlines.append(line)
            figlabels.append(k)
        aa.set_xscale('log')
        aa.set_yscale('log')
        aa.grid(True)
plt.figlegend(figlines, figlabels, loc = 'upper center', ncol=4, labelspacing=0.)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, gridspec_kw=dict(
    hspace=0.1,wspace=0), figsize=(8, 4))
figlines = []
figlabels = []
for i, (k, sub) in enumerate(df.groupby('q')):
    if k == 'DoNothing': continue
    aa = ax.flatten()[order.index(k)]
    aa.set_title(k)
    for a, ssub in sub.groupby('a'):
        n = ssub.index.get_level_values('n')
        t = ssub.t
        line, = aa.plot(n, t, label=labels[a])
        if i == 1:
            figlines.append(line)
            figlabels.append(labels[a])
    aa.set_xscale('log')
    aa.set_yscale('log')
plt.figlegend(figlines, figlabels, loc = 'upper center', ncol=5, labelspacing=0.)
plt.tight_layout()
plt.show()

