import glob
import json
import numpy as np
import matplotlib.pyplot as plt

def go(name, ax, labels, values):
    ax.bar(
            [x for x, y in enumerate(values) if y is not None and np.isfinite(y)],
            [x for x in values if x is not None and np.isfinite(x)], color='#338fff')
    #lim = sorted(values)[-2]*1.5
    lim = sorted(x for x in values if x is not None)[-1]
    for i, (l, t) in enumerate(zip(labels, values)):
        if t is None:
            continue
        ax.text(i, lim/20, f'{l}', rotation=90, ha='center')
        ax.text(i, lim-lim/20, f'{t:.1f}', rotation=90, ha='center', va='top')
    ax.set_ylabel(f'{name} (us/ts)')
    ax.set_ylim(0, lim)
    ax.set_xticks([])

def get(data):
    #labels = list(sorted(data.keys(), key=lambda x: (x!='DoNothing', x)))
    values = [data.get(x, None) for x in all_labels]
    return values # labels, values

fns = glob.glob('*.json')

experiments = 'single batched regular forward'.split()

all_labels = set()
nfigs = 0
for fn in fns:
    data = json.load(open(fn))
    for k in experiments:
        all_labels.update(data.get(k, {}).keys())
        if k in data:
            nfigs += 1
all_labels.remove('BGPQ1')

fig, ax = plt.subplots(figsize=(6, 10), nrows=nfigs, gridspec_kw=dict(hspace=0.2), sharex=True)

counter = 0
for i0, fn in enumerate(fns):
    i0 *= 2
    data = json.load(open(fn))
    title = '{hostname}_{device}'.format(**data['host'])
    ax[counter].set_title(title)
    for k in experiments:
        if k in data:
            #labels,
            values = get(data[k])
            go(k.title(), ax[counter], all_labels, values)
            counter += 1

plt.tight_layout()
plt.savefig('../img/benchmarks.png')
plt.savefig('../img/benchmarks.svg')
plt.show()
