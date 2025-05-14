import glob
import json
import numpy as np
import matplotlib.pyplot as plt

def go(name, ax, labels, values):
    ax.bar(np.arange(len(labels)), values, color='#338fff')
    lim = sorted(values)[-2]*1.5
    for i, (l, t) in enumerate(zip(labels, values)):
        ax.text(i, lim/20, f'{l}', rotation=90, ha='center')
        ax.text(i, lim-lim/20, f'{t:.1f}', rotation=90, ha='center', va='top')
    ax.set_ylabel(f'{name} (us/ts)')
    ax.set_ylim(0, lim)
    ax.set_xticks([])

def get(data):
    labels = list(sorted(data.keys(), key=lambda x: (x!='DoNothing', x)))
    values = [data[x] for x in labels]
    return labels, values

fns = glob.glob('*.json')

fig, ax = plt.subplots(figsize=(6, 10), nrows=2*len(fns), gridspec_kw=dict(hspace=0.2))

for i0, fn in enumerate(fns):
    i0 *= 2
    data = json.load(open(fn))
    title = '{hostname}_{device}'.format(**data['host'])


    ax[i0+0].set_title(title)

    labels, values = get(data['single'])
    go('Single', ax[i0+0], labels, values)
    labels, values = get(data['batched'])
    go('Batch', ax[i0+1], labels, values)
plt.tight_layout()
plt.savefig('../img/benchmarks.png')
plt.savefig('../img/benchmarks.svg')
plt.show()
