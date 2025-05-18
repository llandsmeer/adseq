import os

if '--xla_cpu_use_thunk_runtime=false' not in os.environ.get('XLA_FLAGS', ''):
    print('CONSIDER SETTING')
    print('export XLA_FLAGS=--xla_cpu_use_thunk_runtime=false')

import jax.numpy as jnp
import numpy as np
import jax
import socket
import typing

__all__ = 'get_device_id', 'mkrunner', 'set_backend'

def estimate_backend():
    b = os.environ.get('BACKEND', '').strip()
    if b:
        return b
    hostname = socket.gethostname()
    if hostname == 'Groqhost1':
        return 'groq'
    else:
        return 'jax'

BACKEND: typing.Literal['jax', 'onnxrt', 'groq', 'openvino'] = estimate_backend()

def set_backend(backend: typing.Literal['jax', 'onnxrt', 'groq', 'openvino']):
    global BACKEND
    BACKEND = backend

def get_device_id():
    hostname = socket.gethostname()
    dev = jax.devices()[0]
    device = dev.platform
    hw_version = dev.client.platform_version
    jax_version = str(jax.__version__)
    o = dict(hostname=hostname,
         device=f'{BACKEND}_{device}',
         hw_version=hw_version,
         jax_version=jax_version,
         backend=BACKEND
         )
    return f'{hostname}_{device}_{BACKEND}', o

def _convert_to_tf(f, x, names=None):
    import tensorflow as tf
    from jax.experimental import jax2tf
    f_tf = jax2tf.convert(f, native_serialization=False, enable_xla=False)
    if isinstance(x, tuple):
        sig = [tf.TensorSpec(
            x.shape if hasattr(x, 'shape') else (),
            x.dtype if hasattr(x, 'dtype') else np.array(x).dtype,
            name=f'x{i}' if names is None else names[i]) for i, x in enumerate(x)] # type: ignore
    else:
        sig = [tf.TensorSpec(x.shape, x.dtype, name='x')] # type: ignore
    f_tf = tf.function(f_tf, input_signature=sig, jit_compile=False, autograph=False)
    return f_tf, sig

def _convert_to_onnx(f, x, fn='/tmp/runner.onnx', opset=16, names=None):
    # this doesn't work:
    # from jax2onnx import to_onnx
    # return to_onnx(f, x)
    import tf2onnx
    f_tf, sig = _convert_to_tf(f, x, names)
    onnx = tf2onnx.convert.from_function(f_tf, input_signature=sig, output_path=fn, opset=opset)
    return onnx

def mkrunner_jax(f, x):
    f = jax.jit(f)
    f(x).block_until_ready()
    return lambda: f(x).block_until_ready()

def mkrunner_onnx(f, x):
    import onnxruntime as rt
    _convert_to_onnx(f, x)
    with open('/tmp/runner.onnx', 'rb') as f:
        model = f.read()
    sess = rt.InferenceSession(model)
    arg = dict(x=np.array(x))
    return lambda: sess.run(None, arg)[0]

def mkrunner_groq(f, x):
    from groq.runner import tsp
    _convert_to_onnx(f, x)
    assert 0 == os.system('groq-compiler -o /tmp/runner /tmp/runner.onnx')
    assert 0 == os.system('aa-latest --name runner -i /tmp/runner.aa --output-iop /tmp/runner.iop')
    assert 0 == os.system('iop-utils stats /tmp/runner.iop')
    program = tsp.create_tsp_runner('/tmp/runner.iop')
    x_np = np.array(x)
    k = next(iter(program(x=x_np).keys()))
    return lambda: program(x=x_np)[k]

def mkrunner_openvino(f, x):
    import openvino as ov
    # this doesn't work: convert_model(jax.make_jaxpr(f)(x))
    shape = ov.Shape(x.shape)
    f_tf, sig = _convert_to_tf(f, x)
    del sig
    model = ov.convert_model(f_tf, shape)
    model_c = ov.compile_model(model)
    x_np = np.array(x)
    return lambda: next(iter(model_c(x_np).values()))

def mkrunner_openvino_loop(f_loop, init, xs, unroll=1000):
    # [GPU] Unexpected layout of input memory for if:If_466933 node!
    # Node layout: i32:bfyx:2:nopad
    # Memory layout: i32:bfyx::nopad
    import tqdm
    import openvino as ov
    import jax.tree
    # this doesn't work: convert_model(jax.make_jaxpr(f)(x))
    try:
        assert isinstance(init, tuple)
        structure = jax.tree.structure(init)
        def f_loop_ov(*flat):
            carry, out = jax.lax.scan(f_loop,
                    structure.unflatten(flat[:-2]),
                    (flat[-2], flat[-1]))
            del out
            return { str(i): x for i, x in enumerate(carry) }
        ovsig = tuple(ov.Shape(x.shape if hasattr(x, 'shape') else []) for x in jax.tree.flatten(init)[0]) + \
                (ov.Shape([unroll]), ov.Shape((unroll,) +  xs[0].shape))
        sample = tuple(jax.tree.flatten(init)[0]) + (jnp.arange(unroll), xs[0:unroll])
        sample = tuple(np.array(x) for x in sample)
        init_np = tuple(np.array(x) for x in jax.tree.flatten(init)[0])
        f_tf, sig = _convert_to_tf(f_loop_ov, sample)
        del sig
        model = ov.convert_model(f_tf, ovsig)
        model_c = ov.compile_model(model, 'CPU')
    except Exception as ex:
        return lambda ex=ex: ex
    def runner(xs=xs):
        try:
            carry = init_np
            xs = np.array(xs)
            for i in range(0, len(xs), unroll):
                x = xs[i:i+unroll]
                carry = model_c(carry + (np.arange(i, i+unroll), x))
                carry = tuple(carry[str(i)] for i in range(len(carry)))
            return carry
        except Exception as ex:
            return ex
    return runner

def mkrunner(f, x):
    match BACKEND:
        case 'jax':    return mkrunner_jax(f, x)
        case 'groq':   return mkrunner_groq(f, x)
        case 'onnxrt': return mkrunner_onnx(f, x)
        case 'openvino': return mkrunner_openvino(f, x)
    raise Exception('backend not found')

def mkrunner_onnx_loop(f_loop, init, xs, unroll=10):
    import onnxruntime as rt
    try:
        structure = jax.tree.structure(init)
        def f_loop_unroll(*args):
            carry_in = structure.unflatten([args[i] for i in range(len(args)-2)])
            i = args[-2]
            x = args[-1]
            carry, out = jax.lax.scan(f_loop,
                    carry_in, (i, x), unroll=unroll)
            del out
            carry = jax.tree.flatten(carry)[0]
            return { str(i): x for i, x in enumerate(carry) }
        sample = tuple(jax.tree.flatten(init)[0]) + (jnp.arange(unroll), xs[0:unroll])
        sample = tuple(np.array(x) for x in sample)
        init_np = tuple(np.array(x) for x in jax.tree.flatten(init)[0])
        names = ['C'+str(i) for i in range(len(init_np))] + ['i', 'x']
        _convert_to_onnx(f_loop_unroll, sample, names=names)
        with open('/tmp/runner.onnx', 'rb') as f:
            model = f.read()
        sess = rt.InferenceSession(model)
    except Exception as ex:
        return lambda ex=ex: ex
    def runner(xs=xs):
        try:
            carry = { 'C'+str(i): x for i, x in enumerate(init_np) }
            xs = np.array(xs)
            for i in range(0, len(xs), unroll):
                x = xs[i:i+unroll]
                i = np.arange(i, i+unroll, dtype='int32')
                carry['x'] = x
                carry['i'] = i
                carry = sess.run(None, carry)
                carry = { 'C'+str(i): x for i, x in enumerate(carry) }
            return carry
        except Exception as ex:
            return ex
    return runner

def mkrunner_groq_loop(f_loop, init, xs, unroll=10):
    assert xs.shape[0] % unroll == 0
    from groq.runner import tsp
    structure = jax.tree.structure(init)
    def f_loop_unroll(*args):
        carry_in = structure.unflatten([args[i] for i in range(len(args)-2)])
        i = args[-2]
        x = args[-1]
        carry, out = jax.lax.scan(f_loop,
                carry_in, (i, x), unroll=unroll)
        del out
        carry = jax.tree.flatten(carry)[0]
        return { str(i): x for i, x in enumerate(carry) }
    sample = tuple(jax.tree.flatten(init)[0]) + (jnp.arange(unroll, dtype='int32'), xs[0:unroll])
    sample = tuple(np.array(x) for x in sample)
    init_np = tuple(np.array(x) for x in jax.tree.flatten(init)[0])
    names = ['C'+str(i) for i in range(len(init_np))] + ['i', 'x']
    _convert_to_onnx(f_loop_unroll, sample, names=names)
    assert 0 == os.system('groq-compiler -o /tmp/runner /tmp/runner.onnx')
    assert 0 == os.system('aa-latest --name runner -i /tmp/runner.aa --output-iop /tmp/runner.iop')
    assert 0 == os.system('iop-utils stats /tmp/runner.iop')
    program = tsp.create_tsp_runner('/tmp/runner.iop')

    def recast(x):
        dt = str(x.dtype)
        if dt == 'int64':
            return x.astype('int32')
        if dt == 'float64':
            return x.astype('float32')
        if dt == 'bool':
            return x.astype('int8')
        return x
    init_groq = { 'C'+str(i): recast(x) for i, x in enumerate(init_np) }
    def runner(xs=xs):
        try:
            carry = init_groq
            xs = np.array(xs)
            for i in range(0, len(xs), unroll):
                x = xs[i:i+unroll]
                i = np.arange(i, i+unroll, dtype='int32')
                carry['x'] = x
                carry['i'] = i
                carry = { 'C' + k: v for k, v in program(**carry).items() }
            return carry
        except Exception as ex:
            if 'BufferMismatchException' in repr(ex):
                breakpoint()
            return ex
    return runner

def mkrunner_loop(f_loop, init, xs, **kwargs):
    match BACKEND:
        case 'openvino': return mkrunner_openvino_loop(f_loop, init, xs)
        case 'onnxrt': return mkrunner_onnx_loop(f_loop, init, xs)
        case 'groq': return mkrunner_groq_loop(f_loop, init, xs, unroll=kwargs.get('groq_unroll', None))
    f = lambda stream:jax.lax.scan(
            f=f_loop, # type: ignore
            init=init,
            xs=(jnp.arange(len(stream)), stream)
            )[0][1]
    return mkrunner(f, xs)

