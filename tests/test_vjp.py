import pytest
import jax

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import implementations

# @jax.custom_jvp
# def annotate_grad(x, x_t):
#     del x_t
#     return x
# @annotate_grad.defjvp
# def annotate_grad_jvp(primals, tangents):
#     x, x_t = primals
#     del tangents
#     return x, x_t

@jax.custom_vjp
def annotate_grad(x, y):
    return x
def f_fwd(x, y):
    primal = x
    residual = y,
    return primal, residual
def f_bwd(residual, out_dot):
    y, = residual
    return (out_dot * y, out_dot * y)
annotate_grad.defvjp(f_fwd, f_bwd)

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
def test_pop_reverse_mode_annotated(Q):
    def go(t_spk):
        t_spk = annotate_grad(t_spk, 42.)
        q = Q.init(1, grad=True)
        q = q.enqueue(t_spk)
        return q.pop(10.)[1]
    assert go(10.) == 1
    assert jax.jacrev(go)(10.) == 42

@pytest.mark.parametrize("Q", check)
def test_pop_reverse_mode_equal(Q):
    def go(param):
        t_spk = param ** 2
        q = Q.init(1, grad=True)
        q = q.enqueue(t_spk)
        return q.pop(10.)[1]
    assert jax.jacrev(go)(10.) == jax.jacfwd(go)(10.)
    assert jax.jacrev(go)(20.) != jax.jacfwd(go)(10.)
    assert jax.jacrev(go)(20.) == jax.jacfwd(go)(20.)
