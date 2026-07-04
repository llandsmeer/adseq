import pytest
import jax
import jax.numpy as jnp

from adseq import synapse
from adseq import synapse2

check = [synapse, synapse2]

dt = 0.1
tau_syn = 5.0
alpha = float(jnp.exp(-dt / tau_syn))


@pytest.mark.parametrize("mod", check)
def test_apply_recv_gradient(mod):
    # Fake synapse: spike received once, isyn should rise then decay
    isyn = 0.0
    for i in range(50):
        t = dt * i
        hit = 1.0 if t > 1.0 else 0.0
        isyn = alpha * isyn + float(mod.apply_recv_gradient(hit, tau_syn))
        if t <= 1.0:
            assert isyn == 0.0
    assert isyn > 0.0


@pytest.mark.parametrize("mod", check)
def test_apply_recv_gradient_grad(mod):
    # Fake synapse driven by theta-scaled hit; sum isyn as loss
    def sim(theta):
        isyn = 0.0
        loss = 0.0
        for i in range(50):
            t = dt * i
            hit = theta if t > 1.0 else 0.0
            isyn = alpha * isyn + mod.apply_recv_gradient(hit, tau_syn)
            loss = loss + isyn
        return loss

    g = jax.grad(sim)(1.0)
    assert jnp.isfinite(g)
    assert g > 0  # more theta -> larger hit -> more isyn -> larger sum
