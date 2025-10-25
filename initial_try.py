import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

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

def ratio(Wa,Wb,Pab_p,Pba_p):
    Rab = (Wb/Wa) * (Pba_p/Pab_p)
    # Rba = 1/Rab
    return Rab

def accept_or_reject(R):
    '''
        R stands for both a->b or b->a, but for now just incase this is for a->b
    '''
    return min(1,R)

def Metro(v,T,m,Pab_p = 0.5,Pba_p = 0.5,sigma = 1e4):
    sample = []
    for _ in range(1,1000000):
        if v < 0:
            v = abs(v)
        fv1 = Boltz(m,T,v)
        v_prop = v + np.random.normal(0, sigma)
        fv2 = Boltz(m,T,v_prop)

        R = ratio(fv1,fv2,Pab_p,Pba_p)
        
        proposal = accept_or_reject(R)
        u = np.random.uniform(0, 1)
        if u < proposal:
            v = v_prop
        sample.append(v)
    
    return sample

# ---------------------------- Results ----------------------------

sample = Metro(i_velocity,temperature,mass)

# ---------------------------- Plotting ----------------------------

'''plt.hist(sample, bins=100, density=True, label='Markov Chain Monte Carlo Numerical')

plt.plot(v_vals,boltz, label = 'Analytical')
plt.grid()
plt.title('Maxwell Boltzmann Distribution')
plt.xlabel('Velocity (M/s)')
plt.ylabel('f(v)')
plt.legend()
plt.minorticks_on()
plt.show()'''

import matplotlib.animation as animation

# Prepare figure
fig, ax = plt.subplots()
bins = np.linspace(0, 3e5, 100)

# Plot the analytical curve
ax.plot(v_vals, boltz, 'r-', label='Analytical')
hist_plot = ax.hist([], bins=bins, density=True, alpha=0.6, label='MCMC Samples')[2]
ax.set_xlim(0, 3e5)
ax.set_ylim(0, max(boltz)*1.2)
ax.set_xlabel('Velocity (m/s)')
ax.set_ylabel('f(v)')
ax.legend()
ax.grid(True)

# Function to update the histogram each frame
def update(frame):
    ax.clear()
    ax.plot(v_vals, boltz, 'r-', label='Analytical')
    ax.hist(sample[:frame], bins=bins, density=True, alpha=0.6, label=f'MCMC Samples (N={frame})')
    ax.set_xlim(0, 3e5)
    ax.set_ylim(0, max(boltz)*1.2)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('f(v)')
    ax.legend()
    ax.grid(True)
    return ax,

# Create animation: update every 10,000 samples
frames = np.arange(10000, len(sample), 50000)
ani = animation.FuncAnimation(fig, update, frames=frames, blit=False, repeat=False)

plt.show()
