import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import synapse
import implementations

def plot_sim():
    ts, (trace, spikes) = sim()
    plt.plot(ts, trace)
    plt.show()
    for i, neuron in enumerate(spikes.T):
        print(i)
        idx, = jnp.where(neuron)
        plt.plot(idx, i * jnp.ones_like(idx), 'o')
    plt.show()

def sim():
    n = 23
    dt = 0.025
    tau_syn = 2.
    tau_mem = 10.
    vthres = 1.0
    key = jax.random.PRNGKey(0)
    weight = 0.05 * zero_diagonal(jax.random.normal(key, (n,n)))**2
    delays = 4 + .5*jax.random.normal(key, (n,))
    Q = implementations.SingleSpike
    syn = synapse.mk_synapses(Q, # type: ignore
          delay_ms=delays, dt_ms=dt,
          vthres=vthres, tau_syn_ms=tau_syn, n=n)
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
    print(trace)
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
    plot_sim()
