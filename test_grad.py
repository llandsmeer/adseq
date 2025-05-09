import inspect
import pytest

import jax
import jax.numpy as jnp

import implementations


def test_construct():
    a = implementations.GradientQueue[
            implementations.DoNothing,
            implementations.DoNothing,
            ]

def test_init():
    Q = implementations.GradientQueue[
            implementations.SingleSpike,
            implementations.SingleSpike,
            ]
    Q.init(1)
    jax.jacfwd(lambda x: Q.init(1))(0.)

def test_init_ring():
    Q = implementations.GradientQueue[
            implementations.Ring,
            implementations.Ring,
            ]
    Q.init(1)
    jax.jacfwd(lambda x: Q.init(1))(0.)

def test_enqueue():
    Q = implementations.GradientQueue[
            implementations.DoNothing,
            implementations.DoNothing,
            ]
    q = Q.init(1)
    q = q.enqueue(1)
    q = jax.jacfwd(lambda t_spk: q.enqueue(t_spk))(1.)

def test_pop():
    Q = implementations.GradientQueue[
            implementations.Ring,
            implementations.Ring,
            ]
    q = Q.init(1)
    q = q.enqueue(1)
    def go(t_spk):
        q = q.enqueue(t_spk)
        breakpoint()
        return q.pop(10.)[0]
    q = jax.jacfwd(go)(1.)
