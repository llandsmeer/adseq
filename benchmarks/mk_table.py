import glob
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# DoNothing             0.4269660us/ts
# SingleSpikeKeep       0.6850600us/ts
# SingleSpike           0.9564519us/ts
# LossyRing[4]          1.1763394us/ts
# LossyRing[2]          1.3531685us/ts
# Ring                  1.7534256us/ts
# FIFORing[2]           3.5003304us/ts
# FIFORing[4]           3.5167098us/ts
# FIFORing[8]           3.5546422us/ts


fns = glob.glob('*.json')

experiments = 'single batched regular forward reverse'.split()

dfs = {key: {} for key in experiments}

for fn in fns:
    data = json.load(open(fn))
    host = data['host']
    for key in experiments:
        if key not in data:
            continue
        timings = data[key]
        series = pd.Series(timings)
        dfs[key][f'{host["hostname"]}_{host["device"]}'] = series

def sort_index(x):
    def go(x):
        x, *q = x.split('[')
        o = dict(
                DoNothing=0,
                Ring=10,
                FIFORing=50,
                SortedArray=60,
                BinaryHeap=100,
                BGPQ1=200,
                SingleSpike=20,
                SingleSpikeKeep=30,
                LossyRing=40,
                ).get(x, None)
        if o is not None:
            try:
                n = int(q[0].split(']')[0])
            except Exception as ex:
                print(ex)
                n = 1
            return o*10000 +  n
        print(x)
        return x
    return x.map(go)

def make_pretty(key):
    return dict(
        single='Poisson (single)',
        batched='Poisson (batched)',
        regular='R-SNN',
        forward='R-SNN (forward mode AD)',
        reverse='R-SNN (reverse mode AD)',
    )[key]

header_match = re.compile(r'\\begin\{tabular\}\{(l+)\}')
def add_horizontals(match):
    ls = match.group(1)
    if len(ls) == 1:
        return ls
    else:
        groups = (len(ls) - 1 + 3) // 4
        return r'\begin{tabular}{\textwidth}{' + 'l|' + '|'.join(['rrrr'] * groups) + '|}'


subsets = [
    'single batched regular'.split(),
    'forward reverse'.split(),
    # experiments,
        ]

for subset in subsets:
    dfs2 = {}
    for key in subset:
        print('=='*10, key)
        df = pd.DataFrame(dfs[key])
        pick = {
            'lennartpc_jax_cpu': 'CPU',
            'henkdenktenk_docker_jax_gpu': 'GPU',
            'nonexistent1': 'TPU',
            'Groqhost1_groq_cpu': 'Groq',
        }
        for k in pick:
            if k not in df.columns:
                df[k] = float('nan')
        df = df[list(pick.keys())].rename(pick, axis='columns')
        dfs2[make_pretty(key)] = df

    df = pd.concat(dfs2, axis=1)
    #print(df)

    #df = df.style.format(decimal=',', thousands='.', precision=2)
    df = df.applymap(lambda x: str.format("{:0_.1f}", x) if np.isfinite(x) else '')


    # df = df.apply(lambda row: pd.Series(sorted(row)), axis=1)

    df = df.sort_index(key=sort_index)
    # breakpoint()

    s = df.to_latex(
            na_rep='',
            multicolumn=True,
            bold_rows=False,
            multicolumn_format='c'
            )
    s = header_match.sub(add_horizontals, s)
    s = s.replace('toprule', 'hline')
    s = s.replace('midrule', 'hline')
    s = s.replace('bottomrule', 'hline')
    s = s.replace('{tabular}', '{tabularx}')
    s = s.replace('{c}', '{c|}')
    s = s.replace('_', '')
    print()
    print(s)
    print()
    print()
    print()
    print()
    print()

input()
