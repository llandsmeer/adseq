import inspect
import pytest

import jax
import jax.numpy as jnp

import implementations

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
    implementations.Ring,
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
