import pytest
import jax
import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import implementations
import synapse

check = [
    implementations.SingleSpike,
    implementations.SingleSpikeKeep,
    implementations.Ring,
    implementations.FIFORing,
    implementations.SortedArray,
    implementations.BinaryHeap.sized(5),
    implementations.LossyRing.sized(5),
    ]

@pytest.mark.parametrize("Q", check)
def test_synapses(Q):
    n = 23
    dt = 0.025
    syn = synapse.mk_synapses(Q,
          delay_ms=10, dt_ms=dt,
          vthres=1.0, tau_syn_ms=100.0, n=n)
    f = jax.jit(type(syn).timestep_spike_detect_pre)
    for i in range(1000):
        t = 0.025*i
        v = jnp.ones(n)*t
        vnext = jnp.ones(n)*t + dt
        syn = f(syn, ts=t, v=v, vnext=vnext)
        if t < 1 + 10 - 1.5*0.025:
            assert all(syn.isyn == 0)
        else:
            assert all(syn.isyn > 0)

