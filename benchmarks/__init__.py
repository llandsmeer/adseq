import os

if '--xla_cpu_use_thunk_runtime=false' not in os.environ.get('XLA_FLAGS', ''):
    print('CONSIDER SETTING')
    print('export XLA_FLAGS=--xla_cpu_use_thunk_runtime=false')

import jax
import socket
import typing

__all__ = 'get_device_id', 'mkrunner', 'set_backend'

def estimate_backend():
    hostname = socket.gethostname()
    if hostname == 'Groqhost1':
        return 'groq'
    else:
        return 'jax'

BACKEND: typing.Literal['jax', 'onnxrt', 'groq'] = estimate_backend()

def set_backend(backend: typing.Literal['jax', 'onnxrt', 'groq']):
    global BACKEND
    BACKEND = backend

def get_device_id():
    hostname = socket.gethostname()
    dev = jax.devices()[0]
    device = dev.platform
    hw_version = dev.client.platform_version
    jax_version = str(jax.__version__)
    o = dict(hostname=hostname,
         device=device,
         hw_version=hw_version,
         jax_version=jax_version,
         backend=BACKEND
         )
    return f'{hostname}_{device}', o

def _convert_to_onnx(f, x, fn='/tmp/runner.onnx', opset=16):
    import tensorflow as tf
    import tf2onnx
    from jax.experimental import jax2tf
    f_tf = jax2tf.convert(f, native_serialization=False, enable_xla=False)
    sig = [tf.TensorSpec(x.shape, x.dtype, name='x')] # type: ignore
    f_tf = tf.function(f_tf, input_signature=sig, jit_compile=False, autograph=False)
    onnx = tf2onnx.convert.from_function(f_tf, input_signature=sig, output_path=fn, opset=opset)
    return onnx

def mkrunner_jax(f, x):
    f = jax.jit(f)
    f(x).block_until_ready()
    return lambda: f(x).block_until_ready()

def mkrunner_onnx(f, x):
    import numpy as np
    import onnxruntime as rt
    _convert_to_onnx(f, x)
    with open('/tmp/runner.onnx', 'rb') as f:
        model = f.read()
    sess = rt.InferenceSession(model)
    arg = dict(x=np.array(x))
    return lambda: sess.run(None, arg)[0]

def mkrunner_groq(f, x):
    import numpy as np
    from groq.runner import tsp
    _convert_to_onnx(f, x)
    assert 0 == os.system('groq-compiler -o /tmp/runner /tmp/runner.onnx')
    assert 0 == os.system('aa-latest --name runner -i /tmp/runner.aa --output-iop /tmp/runner.iop')
    assert 0 == os.system('iop-utils stats /tmp/runner.iop')
    program = tsp.create_tsp_runner('/tmp/runner.iop')
    x_np = np.array(x)
    return lambda: program(x=x_np)

def mkrunner(f, x):
    match BACKEND:
        case 'jax':    return mkrunner_jax(f, x)
        case 'groq':   return mkrunner_groq(f, x)
        case 'onnxrt': return mkrunner_onnx(f, x)
