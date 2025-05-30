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
def test_synapse(Q):
    syn = synapse.mk_synapse(Q, delay_ms=10, dt_ms=0.025, vthres=1.0, tau_syn_ms=100.0)
    f = jax.jit(type(syn).timestep_spike_detect_pre)
    for i in range(1000):
        t = 0.025*i
        syn = f(syn, ts=t, v=t, vnext=t+0.025)
        if t < 1 + 10 - 0.025:
            assert syn.isyn == 0
        else:
            assert syn.isyn > 0

@pytest.mark.parametrize("Q", check)
def test_synapse_grad(Q):
    def sim(theta):
        syn = synapse.mk_synapse(Q, delay_ms=1, dt_ms=0.1, vthres=1.0, tau_syn_ms=1.0)
        f = jax.jit(type(syn).timestep_spike_detect_pre)
        loss = 0
        for i in range(40):
            t = 0.1*i
            syn = f(syn, ts=t, v=t*theta, vnext=(t+0.1)*theta)
            goal = t > 2
            loss = loss + (goal - syn.isyn)**2
        return loss
    print(sim(1.0))
    a = jax.grad(sim)(0.5)
    c = jax.grad(sim)(1.5)
    assert jnp.isfinite(a)
    assert jnp.isfinite(c)
    assert a < 0
    assert c > 0

@pytest.mark.parametrize("Q", check)
def test_synapse_grad_wrt_delay(Q):
    def sim(theta):
        syn = synapse.mk_synapse(Q, delay_ms=theta, dt_ms=0.1, vthres=1.0, tau_syn_ms=1.0, max_delay_ms=10)
        f = type(syn).timestep_spike_detect_pre
        loss = 0
        for i in range(40):
            t = 0.1*i
            syn = f(syn, ts=t, v=t, vnext=t+0.1)
            goal = t > 2
            loss = loss + (goal - syn.isyn)**2
        return loss
    g = jax.jit(jax.grad(sim))
    a = g(0.5)
    c = g(2.5)
    assert jnp.isfinite(a)
    assert jnp.isfinite(c)
    assert a < 0
    assert c > 0

