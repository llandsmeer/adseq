import pandas as pd
import matplotlib.pyplot as plt

fns = dict(
    tpu='./caps_t1v-n-9a0ef721-w-0_tpu_jax.json',
    gpu='./caps_gcn97.local.snellius.surf.nl_gpu_jax.json',
    cpu='./caps_spectre_cpu_jax.json'
    )

dfs = {key:
    pd.read_csv(fn, sep=' ', header=None, names=['n', 'q', 't'])
      .apply(lambda x: [x.n, x.q.split('[')[0], x.t], axis='columns', result_type='broadcast')
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
    'BinaryHeap', 'LossyRing', 'FIFORing', 'SortedArray'
       ]

fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, gridspec_kw=dict(
    hspace=0.1,wspace=0), figsize=(8, 4))
figlines = []
figlabels = []
for i, (k, sub) in enumerate(df.groupby('q')):
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

