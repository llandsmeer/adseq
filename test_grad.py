import inspect
import pytest

import jax
import jax.numpy as jnp

import implementations

import synapse


@jax.custom_jvp
def annotate_grad(x, x_t):
    del x_t
    return x
@annotate_grad.defjvp
def annotate_grad_jvp(primals, tangents):
    x, x_t = primals
    del tangents
    return x, x_t


check = [
    implementations.SingleSpike,
    implementations.SingleSpikeKeep,
    implementations.Ring,
    implementations.FIFORing,
    implementations.SortedArray,
    ]
@pytest.mark.parametrize("Q", check)
def test_init(Q):
    Q.init(1)

@pytest.mark.parametrize("Q", check)
def test_enqueue(Q):
    q = Q.init(1, grad=True)
    q = q.enqueue(1)
    q = jax.jacfwd(lambda t_spk: q.enqueue(t_spk))(1.)

@pytest.mark.parametrize("Q", check)
def test_pop(Q):
    def go(t_spk):
        t_spk = annotate_grad(t_spk, 42.)
        q = Q.init(1, grad=True)
        q = q.enqueue(t_spk)
        return q.pop(10.)[1]
    assert go(10.) == 1
    assert jax.jacfwd(go)(10.) == 42

@pytest.mark.parametrize("Q", [x for x in check if x not in [implementations.SingleSpike, implementations.SingleSpikeKeep]])
def test_pop_multi(Q):
    def go(theta):
        t_spk1 = annotate_grad(theta, 42.) + 1.
        t_spk2 = annotate_grad(theta, 24.) + 5.
        q = Q.init(10, grad=True)
        q = q.enqueue(t_spk1)
        q = q.enqueue(t_spk2)
        q, o1 = q.pop(1.)
        q, o2 = q.pop(5.)
        del q
        return o1, o2
    print(go(0))
    assert go(0) == (1, 1)
    assert jax.jacfwd(go)(0.) == (42, 24)

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

