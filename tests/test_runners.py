import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import benchmarks

def test_init():
    f = lambda x: x+1
    x = jnp.array([1])
    assert 2 == benchmarks.mkrunner_jax(f, x)().item()
    assert 2 == benchmarks.mkrunner_onnx(f, x)().item()
