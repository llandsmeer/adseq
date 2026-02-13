import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', False)

import synapse2
import implementations

def exprelr(x): return jnp.where(jnp.isclose(x, 0), 1., x / jnp.expm1(x))
def alpha_m(V): return exprelr(-0.1*V - 4.0)
def alpha_h(V): return 0.07*jnp.exp(-0.05*V - 3.25)
def alpha_n(V): return 0.1*exprelr(-0.1*V - 5.5)
def beta_m(V):  return 4.0*jnp.exp(-(V + 65.0)/18.0)
def beta_h(V):  return 1.0/(jnp.exp(-0.1*V - 3.5) + 1.0)
def beta_n(V):  return 0.125*jnp.exp(-0.0125*V - 0.8125)

@jax.custom_jvp
def superspike(x):
    'doi.dx/10.1162/neco_a_01086'
    return jnp.where(x < 0, 0.0, 1.0)

@superspike.defjvp
def superspike_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    primal_out = jnp.where(x < 0, 0.0, 1.0)
    tangent_out = x_dot / (jnp.abs(x)+1)**2
    return primal_out, tangent_out

dt = 0.025

C_m   =   1.0 ;
E_K   = -77.0 ; E_L   = -53.0 ; E_Na = 50.0
g_Na  = 120.0 ; g_K   =  36.0 ; g_L  =  0.3

@jax.jit
def timestep(t, v, m, n, h, syn, iapp, weight, delay):
    iapp = jnp.array([iapp, weight*syn.isyn])
    I_K = g_K*n**4*(v - E_K)
    I_Na = g_Na*m**3*h*(v - E_Na)
    I_L = g_L*(v - E_L)
    I_total =  I_Na + I_K + I_L - iapp
    vnext = v + dt * (1/C_m)*(-I_total)
    mnext = m + dt * (alpha_m(v)*(1-m) - beta_m(v)*m)
    hnext = h + dt * (alpha_h(v)*(1-h) - beta_h(v)*h)
    nnext = n + dt * (alpha_n(v)*(1-n) - beta_n(v)*n)
    syn = syn.timestep_spike_detect_pre(ts=t, v=v[0], vnext=vnext[0], delay_ms=delay)
    return vnext, mnext, nnext, hnext, syn

def simulate(weight, delay):
    v = jnp.array([-64.64, -64.64])
    m = alpha_m(v) / (alpha_m(v) + beta_m(v))
    h = alpha_h(v) / (alpha_h(v) + beta_h(v))
    n = alpha_n(v) / (alpha_n(v) + beta_n(v))
    syn = synapse2.mk_synapse(
          Q=implementations.FIFORing[4],
          dt_ms=dt,
          vthres=20.0, tau_syn1_ms=0.5, tau_syn2_ms=5.0,
          max_delay_ms=100)
    trace = []
    state0 = v, m, n, h, syn
    ts = jnp.arange(int(round(30 / dt)))
    def body(state, t):
        t = t * dt
        iapp = 20. * ((t > 5) & (t < 7))
        return timestep(t, *state, iapp, weight, delay), (state[0], syn.isyn)
    _, trace = jax.lax.scan(body, state0, ts)
    pre, post = jnp.array(trace[0]).T
    i = trace[1]
    return pre, post, i


delay = 10.
weight = 10.

pre, post, i = simulate(weight, delay)

t = jnp.arange(len(pre)) * dt

g1 = jax.jacfwd(lambda x: simulate(weight, x))(delay)
g2 = jax.jacrev(lambda x: simulate(x, delay))(weight)

g = g1; f = 10
plt.fill_between(jnp.arange(len(post)), post, post+f*0.05*g[1], color='green', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post-f*0.05*g[1], color='red', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post+f*0.04*g[1], color='green', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post-f*0.04*g[1], color='red', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post+f*0.03*g[1], color='green', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post-f*0.03*g[1], color='red', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post+f*0.02*g[1], color='green', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post-f*0.02*g[1], color='red', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post+f*0.01*g[1], color='green', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post-f*0.01*g[1], color='red', edgecolor='none', alpha=0.1)
plt.plot(pre, '--', color='black')
plt.plot(post, color='black')
plt.title('delay')
plt.savefig('delay.svg')
plt.figure()

g = g2; f = 100
plt.fill_between(jnp.arange(len(post)), post, post+f*0.05*g[1], color='green', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post-f*0.05*g[1], color='red', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post+f*0.04*g[1], color='green', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post-f*0.04*g[1], color='red', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post+f*0.03*g[1], color='green', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post-f*0.03*g[1], color='red', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post+f*0.02*g[1], color='green', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post-f*0.02*g[1], color='red', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post+f*0.01*g[1], color='green', edgecolor='none', alpha=0.1)
plt.fill_between(jnp.arange(len(post)), post, post-f*0.01*g[1], color='red', edgecolor='none', alpha=0.1)

plt.title('weight')
plt.plot(pre, '--', color='black')
plt.plot(post, color='black')
plt.show()
