import pytest
import jax

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

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
    implementations.SingleSpikeKeep,
    implementations.Ring,
    implementations.FIFORing,
    implementations.SortedArray,
    implementations.BinaryHeap.sized(7),
    implementations.LossyRing.sized(8),
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
