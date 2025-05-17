#!/home/llandsmeer/repos/llandsmeer/ml_spike_event_queues/benchmarks/env/bin/python
import numpy as np
import optax
import os, sys
import random
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import jit
import jax.tree_util
import jaxley.optimize.transforms as jt

import jaxley as jx
from jaxley.channels import Na, K, Leak
from jaxley.connect import fully_connect

import typing

from jaxley.synapses.synapse import Synapse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import implementations
import synapse

class DelaySynapse(Synapse):
    def __init__(self, name: typing.Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        # queue = implementations.FIFORing.sized(1).init(None, grad=True)
        queue = implementations.SingleSpike.init(None, grad=True)
        queue_list, self.struct = jax.tree.flatten(queue)
        self.struct_size = len(queue_list)
        self.synapse_params = {
            f'{prefix}_tau': 2.,  # ms
            f'{prefix}_delay': 10.,  # ms
            f'{prefix}_weight': 1.,  # ms
        }
        self.synapse_states: dict[str, typing.Any] = {
            f'{prefix}_queue{i}': queue[i]
            for i in range(self.struct_size)
        }
        self.synapse_states[f'{prefix}_vprev'] = 0.
        self.synapse_states[f'{prefix}_isyn'] = 0.
        self.synapse_states[f'{prefix}_ts'] = 0.

    def update_states(
        self,
        states: dict,
        delta_t: float,
        pre_voltage: float,
        post_voltage: float,
        params: dict,
    ) -> dict:
        prefix = self._name
        queues = self.struct.unflatten([
            states[f'{prefix}_queue{i}']
            for i in range(self.struct_size)
            ])
        ts = states[f'{prefix}_ts']
        vthres = 1.0
        delay_ms = params[f'{prefix}_delay']
        tau_syn_ms = params[f'{prefix}_tau']
        def timestep(ts, queue, isyn, v, vnext, delay_ms, tau_syn_ms):
            alpha = jnp.exp(-delta_t / tau_syn_ms) # inefficient
            tpost = synapse.spike_detect(delta_t, ts, vthres, v, vnext, delay_ms)
            queue = jax.lax.cond(tpost != -1, # must be a better solution
                 lambda: queue.enqueue(synapse.time_to_timestep_keep_gradient(tpost, delta_t)), # type: ignore
                 lambda: queue)
            queue, post_hit = queue.pop(synapse.time_to_timestep_keep_gradient(ts, delta_t))
            isyn = alpha * isyn + \
                   synapse.apply_recv_gradient(post_hit, tau_syn_ms)
            return (queue, isyn)
        vprev = states[f'{prefix}_vprev']
        isyn = states[f'{prefix}_isyn']
        queues, isyn = jax.vmap(timestep)(
                ts, #ugly
                queues, isyn,
                vprev, pre_voltage, delay_ms, tau_syn_ms)
        queue_parts, _struct = jax.tree.flatten(queues)
        state_out: dict[str, typing.Any] = {
            f'{prefix}_queue{i}': queue_parts[i]
            for i in range(self.struct_size)
        }
        state_out[f'{prefix}_vprev'] = pre_voltage
        state_out[f'{prefix}_isyn'] = isyn
        state_out[f'{prefix}_ts'] = states[f'{prefix}_ts'] + delta_t
        return state_out

    def compute_current(
        self, states: dict, pre_voltage: float, post_voltage: float, params: dict
    ) -> float:
        prefix = self._name
        return -0.01 * states[f'{prefix}_isyn'] * params[f'{prefix}_weight']

def sim():
    num_cells = 11
    delays = jnp.array(2+3*np.random.random(num_cells*(num_cells-1)))
    weights = jnp.array(5*np.random.random(num_cells*(num_cells-1)))
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=4)
    cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1, 2, 2])
    i_delay = 3.0  # ms
    i_amp = 0.05  # nA
    i_dur = 2.0  # ms
    dt = 0.025  # ms
    t_max = 50.0  # ms
    net = jx.Network([cell for _ in range(num_cells)])
    pre = net.cell(range(num_cells))
    post = net.cell(range(num_cells))
    fully_connect(pre, post, DelaySynapse(), True)
    idx = np.arange(num_cells)
    net.select(edges=idx*num_cells+idx).set('DelaySynapse_delay', 0)
    net.select(edges=idx*num_cells+idx).set('DelaySynapse_weight', 0)
    nonself = np.array([i*num_cells + j for i in range(num_cells) for j in range(num_cells) if i != j])
    net.select(edges=nonself).set('DelaySynapse_delay', delays)
    net.select(edges=nonself).set('DelaySynapse_weight', weights)
    net.select(edges=nonself).make_trainable('DelaySynapse_delay')
    # net.select(edges=nonself).make_trainable('DelaySynapse_weight')
    net.insert(Na())
    net.insert(K())
    net.insert(Leak())
    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    net.delete_stimuli()
    for stim_ind in range(10):
        net.cell(stim_ind).branch(0).loc(0.0).stimulate(current)
    net.delete_recordings()
    net.cell(range(11)).branch(0).loc(0.0).record()
    parameters = net.get_parameters()
    # Define parameter transform and apply it to the parameters.
    transform = jx.ParamTransform([
        {'DelaySynapse_delay':  jt.SigmoidTransform(1.0, 5.0)},
        # {'DelaySynapse_weight': jt.SigmoidTransform(0.0, 5.0)}
    ])
    def loss(opt_params):
        params = transform.forward(opt_params)
        s = jx.integrate(net, delta_t=dt, params=params)
        return s.mean()
    opt_params = transform.inverse(parameters)
    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(opt_params)
    g = jax.jit(jax.value_and_grad(loss, argnums=0))
    OLD = transform.forward(opt_params)
    for _ in range(200):
        loss, gradient = g(opt_params)
        updates, opt_state = optimizer.update(gradient, opt_state)
        opt_params = optax.apply_updates(opt_params, updates)
        print(loss)
    NEW = transform.forward(opt_params)
    print(NEW[0]['DelaySynapse_delay'] - OLD[0]['DelaySynapse_delay'])
    s_old = jx.integrate(net, delta_t=dt, params=OLD)
    s_new = jx.integrate(net, delta_t=dt, params=NEW)
    plt.plot(s_old.T, color='black')
    plt.plot(s_new.T-100, color='black')
    plt.savefig('./img/delay_training.png')
    plt.show()

if __name__ == '__main__':
    sim()
