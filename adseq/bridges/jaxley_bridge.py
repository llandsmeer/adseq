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

from .. import implementations
from .. import synapse2
from ..implementations import SingleSpike

__all__ = 'DelaySynapse','SingleSpike', 'FIFORing', 'FIFORing2', 'FIFORing3', 'FIFORing4', 'FIFORing5'

FIFORing = implementations.FIFORing
FIFORing2 = FIFORing.sized(2)
FIFORing3 = FIFORing.sized(3)
FIFORing4 = FIFORing.sized(4)
FIFORing5 = FIFORing.sized(5)



class DelaySynapse(Synapse):
    def __init__(self, name: typing.Optional[str] = None, Q=SingleSpike, vthres=10.):
        super().__init__(name)
        prefix = self._name
        # queue = implementations.FIFORing.sized(1).init(None, grad=True)
        queue = Q.init(None, grad=True)
        queue_list, self.struct = jax.tree.flatten(queue)
        self.struct_size = len(queue_list)
        self.synapse_params = {
            f'{prefix}_tau1': 0.5,  # ms
            f'{prefix}_tau2': 2,  # ms
            f'{prefix}_delay': 20.,  # ms
            f'{prefix}_weight': 0.01,
        }
        self.synapse_states: dict[str, typing.Any] = {
            f'{prefix}_queue{i}': queue[i]
            for i in range(self.struct_size)
        }
        self.synapse_states[f'{prefix}_vprev'] = 0.
        self.synapse_states[f'{prefix}_isyn1'] = 0.
        self.synapse_states[f'{prefix}_isyn2'] = 0.
        self.synapse_states[f'{prefix}_ts'] = 0.
        self.vthres = vthres

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
        delay_ms = params[f'{prefix}_delay']
        tau1_syn_ms = params[f'{prefix}_tau1']
        tau2_syn_ms = params[f'{prefix}_tau2']
        def timestep(ts, queue, isyn1, isyn2, v, vnext, delay_ms, tau1_syn_ms, tau2_syn_ms):
            alpha = jnp.exp(-delta_t / tau1_syn_ms) # inefficient
            beta = jnp.exp(-delta_t / tau2_syn_ms) # inefficient
            tpost = synapse2.spike_detect(delta_t, ts, self.vthres, v, vnext, delay_ms)
            queue = jax.lax.cond(tpost != -1, # must be a better solution
                 lambda: queue.enqueue(synapse2.time_to_timestep_keep_gradient(tpost, delta_t)), # type: ignore
                 lambda: queue)
            queue, post_hit = queue.pop(synapse2.time_to_timestep_keep_gradient(ts, delta_t))
            jump1 = synapse2.apply_recv_gradient(post_hit, tau1_syn_ms)
            jump2 = synapse2.apply_recv_gradient(post_hit, tau2_syn_ms)
            isyn1 = alpha * isyn1 + jump1
            isyn2 = beta * isyn2 + jump2
            return (queue, isyn1, isyn2)
        vprev = states[f'{prefix}_vprev']
        isyn1 = states[f'{prefix}_isyn1']
        isyn2 = states[f'{prefix}_isyn2']
        queues, isyn1, isyn2 = jax.vmap(timestep)(
                ts, #ugly
                queues, isyn1, isyn2,
                v=vprev, vnext=pre_voltage, delay_ms=delay_ms, tau1_syn_ms=tau1_syn_ms, tau2_syn_ms=tau2_syn_ms)
        queue_parts, _struct = jax.tree.flatten(queues)
        state_out: dict[str, typing.Any] = {
            f'{prefix}_queue{i}': queue_parts[i]
            for i in range(self.struct_size)
        }
        state_out[f'{prefix}_vprev'] = pre_voltage
        # jax.debug.print('{} {} {}', pre_voltage, states[f'{prefix}_vprev'], state_out[f'{prefix}_vprev'])
        state_out[f'{prefix}_isyn1'] = isyn1
        state_out[f'{prefix}_isyn2'] = isyn2
        state_out[f'{prefix}_ts'] = states[f'{prefix}_ts'] + delta_t
        return state_out

    def compute_current(
        self, states: dict, pre_voltage: float, post_voltage: float, params: dict
    ) -> float:
        prefix = self._name
        tau_syn1_ms = params[f'{prefix}_tau1']
        tau_syn2_ms = params[f'{prefix}_tau2']
        t_peak = (tau_syn1_ms * tau_syn2_ms / (tau_syn2_ms - tau_syn1_ms) * jnp.log(tau_syn2_ms / tau_syn1_ms))
        denom = (jnp.exp(-t_peak / tau_syn2_ms) - jnp.exp(-t_peak / tau_syn1_ms))
        return -(states[f'{prefix}_isyn2']-states[f'{prefix}_isyn1'])/denom * params[f'{prefix}_weight']
