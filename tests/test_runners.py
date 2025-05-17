import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import benchmarks

f = lambda x: x+1
x = jnp.array([1])

def test_jax():
    assert 2 == benchmarks.mkrunner_jax(f, x)().item()

def test_onnx():
    assert 2 == benchmarks.mkrunner_onnx(f, x)().item()

def test_groq():
    assert 2 == benchmarks.mkrunner_groq(f, x)().item()

def test_openvino():
    assert 2 == benchmarks.mkrunner_openvino(f, x)().item()
