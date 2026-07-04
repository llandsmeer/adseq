import pytest
import jax
import jax.numpy as jnp

from adseq import implementations
from adseq import synapse

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
    @jax.jit
    def sim(theta):
        dt = 0.01
        syn = synapse.mk_synapse(Q, delay_ms=1, dt_ms=dt, vthres=1.0, tau_syn_ms=1.0)
        f = jax.jit(type(syn).timestep_spike_detect_pre)
        def step(carry, i):
            syn, v = carry

            t = dt * i
            vnext = v + dt * theta
            syn = f(syn, ts=t, v=v, vnext=vnext)

            return (syn, vnext), None

        (syn, _), _ = jax.lax.scan(step, (syn, 0.0), jnp.arange(400))
        return (syn.isyn - 0.2)**2
    print('0.4', sim(0.4))
    print('0.5', sim(0.5))
    print('0.6', sim(0.6))
    print('1.0', sim(1.0))
    print('1.4', sim(1.4))
    print('1.5', sim(1.5))
    print('1.6', sim(1.6))
    dparam = 0.01
    a_estim = (sim(0.5+dparam) - sim(0.5-dparam)) / (2*dparam)
    a = jax.grad(sim)(0.5)
    c = jax.grad(sim)(1.5)
    print('a', a)
    print('a_estim', a_estim)
    assert jnp.isfinite(a)
    assert jnp.isfinite(c)
    assert a < 0
    assert c > 0
    assert abs(a - a_estim) < abs(a_estim) * 0.1

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

