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



# ---------------------------- Plotting ----------------------------

plt.plot(v_vals,boltz, label = 'Analytical')
plt.grid()
plt.title()
plt.xlabel()
plt.ylabel()
plt.legend()
plt.minorticks_on()
plt.show()
