import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
# Intial parameters

i_velocity = 1e6
mass = sc.m_e
temperature = 280 # Kelvin

# ---------------------------- Analytical ----------------------------

def Boltz(m,T,v):
    '''
        m = Mass
        T = Temperature
        v = initial velocity
    '''
    return ( ((m)/(2 * np.pi * sc.k * T)) ** (3/2) ) * (4*np.pi*v**2) * np.exp((-m * v**2)/(2 * sc.k * T))

v_vals = np.arange(0,0.3e6,1)
boltz = []
for i in v_vals:
    boltz.append(Boltz(mass,temperature,i))

# ---------------------------- Metropolis Hastings ----------------------------

def ratio(fa,fb,xa,xb):
    Rab = (fb/fa) * (xa/xb)
    # Rba = 1/Rab
    return Rab

def accept_or_reject(R):
    '''
        R stands for both a->b or b->a, but for now just incase this is for a->b
    '''
    return min(1,R)

def Metro(v,T,m,lamb):
    sample = []
    for _ in range(1,1000000):
        if v < 0:
            v = abs(v)
        fv1 = Boltz(m,T,v)
        v_prop = v/lamb + np.random.random() * (lamb * v - v/lamb)
        fv2 = Boltz(m,T,v_prop)
        R = ratio(fv1,fv2,v,v_prop)
        acceptance_prob = accept_or_reject(R)
        u = np.random.random()
        if u < acceptance_prob:
            v = v_prop
        sample.append(v)
    
    return sample

# ---------------------------- Results ----------------------------

sample = Metro(i_velocity,temperature,mass,2)

# ---------------------------- Plotting ----------------------------

plt.hist(sample, bins=100, density=True, label='Markov Chain Monte Carlo Numerical')

plt.plot(v_vals,boltz, label = 'Analytical', color = 'r')
plt.grid()
plt.title('Maxwell Boltzmann Distribution')
plt.xlabel('Velocity (M/s)')
plt.ylabel('f(v)')
plt.legend()
plt.xlim(0, 3e5)
plt.ylim(0, max(boltz)*1.2)

plt.minorticks_on()
plt.show()
'''
fig, ax = plt.subplots()
bins = np.linspace(0, 3e5, 100)

ax.plot(v_vals, boltz, 'r-', label='Analytical')
hist_plot = ax.hist([], bins=bins, density=True, alpha=0.6, label='MCMC Samples')[2]
ax.set_xlim(0, 3e5)
ax.set_ylim(0, max(boltz)*1.2)
ax.set_xlabel('Velocity (m/s)')
ax.set_ylabel('f(v)')
ax.set_title('Maxwell Boltzmann Distribution')

ax.legend()
ax.grid(True)

def update(frame):
    ax.clear()
    ax.plot(v_vals, boltz, 'r-', label='Analytical')
    ax.hist(sample[:frame], bins=bins, density=True, alpha=0.6, label=f'MCMC Samples (N={frame})')
    ax.set_xlim(0, 3e5)
    ax.set_ylim(0, max(boltz)*1.2)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('f(v)')
    ax.set_title('Maxwell Boltzmann Distribution')

    ax.legend()
    ax.grid(True)
    return ax,

frames = np.arange(10000, len(sample), 50000)
ani = animation.FuncAnimation(fig, update, frames=frames, blit=False, repeat=False)

'''
for i in range(1,10001,50)
    sample = Metro(i_velocity, temperature, mass,i)
    plt.hist(sample, bins=100, density=True, label='Markov Chain Monte Carlo Numerical')

    plt.plot(v_vals,boltz, label = 'Analytical', color = 'r')
    plt.grid()
    plt.title(f'Maxwell Boltzmann Distribution for $\\lambda = ${i}')
    plt.xlabel('Velocity (M/s)')
    plt.ylabel('f(v)')
    plt.legend()
    plt.xlim(0, 3e5)
    plt.ylim(0, max(boltz)*1.2)
    plt.minorticks_on()


    #import os, create file path and save to a path
