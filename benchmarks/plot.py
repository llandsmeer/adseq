import glob
import json
import numpy as np
import matplotlib.pyplot as plt

def go(ax, labels, values):
    ax.bar(np.arange(len(labels)), values, color='#338fff')
    lim = values[-2]*1.5
    for i, (l, t) in enumerate(zip(labels, values)):
        ax.text(i, lim/20, f'{l}', rotation=90, ha='center')
        ax.text(i, lim-lim/20, f'{t:.1f}', rotation=90, ha='center', va='top')
    ax.set_ylabel("us/ts")
    ax.set_ylim(0, lim)
    ax.set_xticks([])

fns = glob.glob('*.json')

fig, ax = plt.subplots(nrows=2*len(fns))

for i0, fn in enumerate(fns):
    data = json.load(open(fn))
    title = '{hostname}_{device}'.format(**data['host'])

    single_labels = list(data["single"].keys())
    single_values = list(data["single"].values())
    batched_labels = list(data["batched"].keys())
    batched_values = list(data["batched"].values())

    ax[i0+0].set_title(title)

    go(ax[i0+0], single_labels, single_values)
    go(ax[i0+1], batched_labels, batched_values)
plt.tight_layout()
plt.savefig('../img/benchmarks.png')
plt.savefig('../img/benchmarks.svg')
plt.show()
