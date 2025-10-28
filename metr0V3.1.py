import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from matplotlib.animation import FuncAnimation

# Intial parameters

i_velocity = 1e6
mass = sc.m_e
temperature = 280  # Kelvin
saveF = False
saveT = True


# ---------------------------- Analytical ----------------------------


def Boltz(m, T, v):
    """
    m = Mass
    T = Temperature
    v = initial velocity
    """
    return (
        (((m) / (2 * np.pi * sc.k * T)) ** (3 / 2))
        * (4 * np.pi * v**2)
        * np.exp((-m * v**2) / (2 * sc.k * T))
    )


v_vals = np.arange(0, 0.3e6, 1)
boltz = []
for i in v_vals:
    boltz.append(Boltz(mass, temperature, i))

# ---------------------------- Metropolis Hastings ----------------------------


def ratio(fa, fb, xa, xb):
    Rab = (fb / fa) * (xa / xb)
    # Rba = 1/Rab
    return Rab


def accept_or_reject(R):
    """
    R stands for both a->b or b->a, but for now just incase this is for a->b
    """
    return min(1, R)


def Metro(v, T, m, lamb, N=1000000):
    sample = []
    for _ in range(N):
        if v < 0:
            v = abs(v)
        fv1 = Boltz(m, T, v)
        v_prop = v / lamb + np.random.random() * ((lamb * v) - (v / lamb))
        fv2 = Boltz(m, T, v_prop)
        R = ratio(fv1, fv2, v, v_prop)
        acceptance_prob = accept_or_reject(R)
        u = np.random.random()
        if u < acceptance_prob:
            v = v_prop
        sample.append(v)

    return sample


# ---------------------------- Results ----------------------------
""" Animation 
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
plt.close()


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


"""

if saveF:  # Saving gifs
    path = r"C:\Users\fowar\OneDrive\Desktop\Folder\university\picgif"
    for i in range(1, 10001, 50):
        sample = Metro(i_velocity, temperature, mass, i)
        plt.hist(
            sample, bins=100, density=True, label="Markov Chain Monte Carlo Numerical"
        )

        plt.plot(v_vals, boltz, label="Analytical", color="r")
        plt.grid()
        plt.title(f"Maxwell Boltzmann Distribution for $\\lambda = ${i}")
        plt.xlabel("Velocity (M/s)")
        plt.ylabel("f(v)")
        plt.legend()
        plt.xlim(0, 3e5)
        plt.ylim(0, max(boltz) * 1.2)
        plt.minorticks_on()

        filename = f"MBlambda_{i}.png"
        file_path = os.path.join(path, filename)

        plt.savefig(file_path)
        plt.close()

        # import os, create file path and save to a path

aug_vp = (2 * 1 / np.sqrt(np.pi)) * np.sqrt((2 * sc.k * temperature) / mass)


def Metro_expectation(v, T, m, lamb, N=1000000, C=True):
    """
    The function g_o(x) that was used to perform the calculation for the integral and for C is e^-x
    """
    if C:
        sample = Metro(v, T, m, lamb, N)
        running = 0
        running_array = []
        steps = []
        for i in range(1, N+1):
            v_n = sample[i - 1]
            running += v_n
            running_array.append(running / i)
            steps.append(i)
        return running_array, steps
    else:
        # Int
        sample = Metro(v, T, m, lamb, N)
        V = np.array(sample) / v # Dividing all numbers by expected v or the v we start with ot try to 'normalise' so e^-x term doesn't explode
        running = 0
        running_array = []
        steps = []

        # C
        run_C = 0
        run_C_array = []
        for i in range(1, N+1):
            # Calculating C
            if i < 10000:
                continue
            if i >= 10000:
                denominator = np.exp(-1 * V[i - 1]) / Boltz(m, T, V[i - 1])
                run_C += denominator
                #print(f"Summing term: {run_C}")
                M_reciprocal_run_C = 1 / ((1 / i) * run_C)
                #print(f"C: {M_reciprocal_run_C}")
                run_C_array.append(M_reciprocal_run_C)

                v_n = V[i - 1]
                running += v_n
                running_array.append(M_reciprocal_run_C * (running / i))
                steps.append(i)
    return running_array, steps, run_C_array


def Metro_expectation2(v, T, m, lamb, N=1000000, C=True):
    """
    The function g_o(x) that was used to perform the calculation of C is sin(x)/x
    """
    if C:
        sample = Metro(v, T, m, lamb, N)
        running = 0
        running_array = []
        steps = []
        for i in range(1, N):
            v_n = sample[i - 1]
            running += v_n
            running_array.append(running / i)
            steps.append(i)
        return running_array, steps
    else:
        # Int
        sample = Metro(v, T, m, lamb, N)
        running = 0
        running_array = []
        steps = []

        # C
        run_C = 0
        run_C_array = []

        for i in range(1, N):
            # Calculating C
            denominator = np.sin(sample[i - 1]) / (
                sample[i - 1] * Boltz(m, T, sample[i - 1])
            )
            run_C += denominator
            M_reciprocal_run_C = (np.pi / 2) * i * (1 / run_C)
            run_C_array.append(M_reciprocal_run_C)

            v_n = sample[i - 1]
            running += v_n
            running_array.append(M_reciprocal_run_C * (running / i))
            steps.append(i)
    return running_array, steps, run_C_array


if saveF:
    path = r"C:\Users\fowar\OneDrive\Desktop\Folder\university\picgif2"
    for i in np.linspace(0.5, 6, 100):
        run, step = Metro_expectation(aug_vp, temperature, mass, i, 1000000)

        plt.plot(step, run)
        plt.plot(step, [aug_vp] * len(step), "r--", label="Analytical")
        plt.grid()
        plt.title(
            f"Velocity Expectation: Maxwell Boltzmann Distribution with $C = 1$, $\\lambda = ${round(i,3)}"
        )
        plt.xlabel("Step")
        plt.ylim(103000, 105000)
        plt.ylabel("$<v>$")
        plt.legend()
        plt.minorticks_on()

        filename = f"EMBlambda_{round(i,3)}.png"
        file_path = os.path.join(path, filename)
        print(f"graph done!")

        plt.savefig(file_path)
        plt.close()
# sample = Metro(aug_vp,temperature,mass,2)
#run, step, C = Metro_expectation(aug_vp, temperature, mass, 1.8, 1000000, False)
run, step = Metro_expectation(aug_vp, temperature, mass, 2, 100000000, True)

#plt.plot(step, C)
#plt.plot(step, [1] * len(step), "r--", label="Analytical")
plt.plot(step,run)
plt.ylim(aug_vp-20, aug_vp +20)
# plt.plot(step,run)
plt.plot(step,[aug_vp] * len(step),'r--', label = 'Analytical')
# plt.plot(v_vals,np.array(boltz) * 1e6, label = 'Analytical', color = 'r')
plt.grid()
plt.title('Velocity Expectation: Maxwell Boltzmann Distribution (C=1)')
#plt.title("Velocity Expectation: Maxwell Boltzmann Distribution C calculation")
plt.xlabel("Step")
plt.ylabel("$<v>$")
plt.legend()
plt.minorticks_on()
plt.show()