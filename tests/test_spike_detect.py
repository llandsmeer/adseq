import pytest
import jax
import jax.numpy as jnp

from adseq import synapse
from adseq import synapse2

check = [synapse, synapse2]


@pytest.mark.parametrize("mod", check)
def test_spike_detect(mod):
    # v = t grows linearly (theta=1), crosses vthres=1.0 at t=1.0
    dt = 0.025
    delay = 2.0
    for i in range(200):
        t = dt * i
        tpost = mod.spike_detect(dt, t, 1.0, t, t + dt, delay)
        if t < 1.0 - dt:
            assert tpost == -1.0
        if tpost > 0:
            assert tpost == pytest.approx(t + delay)


@pytest.mark.parametrize("mod", check)
def test_spike_detect_grad(mod):
    # v = t * theta; spike fires around t = 1/theta; tpost = t + delay
    # loss = (tpost - ttgt)**2 on the spike step
    ttgt = 2.0
    delay = 1.0

    def sim(theta):
        loss = 0
        for i in range(40):
            t = 0.1 * i
            tpost = mod.spike_detect(0.1, t, 1.0, t * theta, (t + 0.1) * theta, delay)
            loss = loss + jnp.where(tpost > 0, (tpost - ttgt) ** 2, 0.0)
        return loss

    a = jax.grad(sim)(0.5)   # spike too late  (tpost > ttgt) -> grad < 0
    c = jax.grad(sim)(1.5)   # spike too early (tpost < ttgt) -> grad > 0
    assert jnp.isfinite(a)
    assert jnp.isfinite(c)
    assert a < 0
    assert c > 0
