#!/usr/bin/env python
import json
import time

import tqdm

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import synapse
import implementations

def get_device_id():
    import socket
    hostname = socket.gethostname()
    dev = jax.devices()[0]
    device = dev.platform
    hw_version = dev.client.platform_version
    jax_version = str(jax.__version__)
    o = dict(hostname=hostname,
         device=device,
         hw_version=hw_version,
         jax_version=jax_version)
    return f'{hostname}_{device}', o

def main():
    results = {
            'regular': {},
            'forward': {}
            }
    dev_name, results['host'] = get_device_id()
    print('=== regular ===')
    results['regular'].update(benchmark_regular())
    print('=== forward ===')
    results['forward'].update(benchmark_grad(jax.jacfwd))
    with open(f'benchmarks/{dev_name}_grad.json', 'w') as f:
        json.dump(results, f)
    # print('=== reverse ===')
    # benchmark_grad(jax.jacrev)

qs = [
    implementations.DoNothing,
    implementations.SingleSpike,
    implementations.SingleSpikeKeep,
    implementations.FIFORing.sized(2),
    implementations.FIFORing.sized(3),
    implementations.FIFORing.sized(4),
    implementations.SortedArray.sized(2),
    implementations.SortedArray.sized(3),
    implementations.SortedArray.sized(4),
    implementations.Ring,
    ]

def benchmark_regular():
    key = jax.random.PRNGKey(0)
    n = 10
    weight = jnp.sqrt(23)/jnp.sqrt(n) * 0.05 * zero_diagonal(jax.random.normal(key, (n,n)))**2
    out = {}
    for Q in qs:
        t = jax.jit(lambda w: sim(n, w, Q=Q)[1][1].sum())
        t(weight).block_until_ready()
        deltas = []
        for _ in range(5):
            a = time.time()
            t(weight).block_until_ready()
            b = time.time()
            deltas.append(b - a)
        tmean = jnp.mean(jnp.array(deltas))
        out[Q.__name__] = float(tmean) / 10000 * 1e6
        print(Q.__name__.ljust(20), tmean, 'seconds')
    return out

def benchmark_grad(jac=jax.jacfwd):
    key = jax.random.PRNGKey(0)
    n = 10
    weight = jnp.sqrt(23)/jnp.sqrt(n) * 0.05 * zero_diagonal(jax.random.normal(key, (n,n)))**2
    out = {}
    for Q in qs:
        t = jax.jit(jac(lambda w: sim(n, w, Q=Q)[1][1].sum()))
        t(weight).block_until_ready()
        deltas = []
        for _ in range(5):
            a = time.time()
            t(weight).block_until_ready()
            b = time.time()
            deltas.append(b - a)
        tmean = jnp.mean(jnp.array(deltas))
        out[Q.__name__] = float(tmean) / 10000 * 1e6
        print(Q.__name__.ljust(20), tmean, 'seconds')
    return out

def plot_sim(n):
    'plot_sim(100)'
    ts, (trace, spikes) = sim(n)
    plt.plot(ts, spikes.sum(1))
    plt.ylim(0, 30)
    plt.show()
    plt.plot(ts, trace)
    plt.show()
    for i, neuron in enumerate(tqdm.tqdm(spikes.T)):
        idx, = jnp.where(neuron)
        plt.plot(idx, i * jnp.ones_like(idx), 'o')
    plt.show()

def sim(
    n = 23,
    weight: None | jax.Array = None,
    Q = implementations.SingleSpike
        ):
    dt = 0.025
    tau_syn = 2.
    tau_mem = 10.
    vthres = 1.0
    key = jax.random.PRNGKey(0)
    if weight is None:
        weight = jnp.sqrt(23)/jnp.sqrt(n) * 0.05 * zero_diagonal(jax.random.normal(key, (n,n)))**2
    delays = 4 + .5*jax.random.normal(key, (n,))
    syn = synapse.mk_synapses(Q, # type: ignore
          delay_ms=delays, dt_ms=dt,
          vthres=vthres, tau_syn_ms=tau_syn, n=n,
          max_delay_ms=7
          )
    syn_step = jax.jit(type(syn).timestep_spike_detect_pre)
    v = jnp.zeros(n)
    state = v, syn
    def step(state, t):
        v, syn = state
        isyn = (weight @ jnp.roll(syn.isyn, 1)).at[:].add(1. * (t < 2))
        # ring:
        # isyn = (weight * jnp.roll(syn.isyn, 1)).at[0].add(1. * (t < 2))
        vnext, s = lif_step(v, isyn, tau_mem, dt, vthres)
        syn = syn_step(syn, ts=t, v=v, vnext=vnext)
        state = vnext, syn
        return state, (v, s)
    ts = jnp.arange(10000) * dt
    _, trace = jax.lax.scan(step, state, xs=ts)
    return ts, trace

@jax.custom_jvp
def superspike(x):
    'doi.dx/10.1162/neco_a_01086'
    return jnp.where(x < 0, 0.0, 1.0)

@superspike.defjvp
def superspike_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    primal_out = jnp.where(x < 0, 0.0, 1.0)
    tangent_out = x_dot / (jnp.abs(x)+1)**2
    return primal_out, tangent_out

def lif_step(U: jax.Array, I: jax.Array, tau_mem: float, dt: float, vth: float =1):
    S = superspike(U - vth)
    beta = jnp.exp(-dt/tau_mem)
    U_next = (1 - S) * (beta * U + I*dt)
    return U_next, S

def zero_diagonal(arr):
    n = arr.shape[0]
    return arr * (1 - jnp.eye(n))

if __name__ == '__main__':
    main()
