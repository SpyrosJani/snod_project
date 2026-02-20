import numpy as np
import sys
import matplotlib.pylab as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
np.set_printoptions(threshold=sys.maxsize)

# ------------------------------------------------- Definition of functions --------------------------------------------------
"""
This function defines the S-NOD dynamical system

Due to the solve_ivp function, it takes as the first argument the timestamps used in the numerical integration, 
even though it is not explicitly used in the system
"""
def system(t, vect, a, b, d, eps, k, k_s, m_0):
    z, s = vect[0], vect[1]
    z_dot = -d*z + np.tanh((k*z**2 + m_0 - s)*a*z + b)
    s_dot = eps*(-s + k_s * z**4)
    return [z_dot, s_dot]

"""
This fuction defines the z-nullcline (z' = 0). Its definition stems from simply solving z' = f(z, s) 
for the s-variable
"""
def z_null(z, a, b, d, k, m_0):
    return k*z**2 + m_0 - np.arctanh(d*z)/(a*z) + b/(a*z)

"""
Defines the function for finding the equilibrium of the system in the z-coordinate (z'=0)

It comes from substituting s = k_s*z**4, which is the condition for the equilibrium 
in the s-coordinate (s' = 0)

If the input parameter z is indeed an equilibrium, this function should return 0 (or something very close to 0)
"""
def equilibirum(z, a, b, d, k, m_0, k_s): 
    return -d*z + np.tanh((k*z**2 + m_0 - k_s*z**4)*a*z + b)

"""
Defines the function for finding the intersection of the nullclines

The roots of this function correspond to the z-coordinates of the equilibria of the system
"""
def nullcline_intersection(z, a, b, d, k, k_s, m_0):
    return k_s * z**4 - z_null(z, a, b, d, k, m_0)



# ------------------------------------------------- Initialisation --------------------------------------------------------------

"""
random.seed() is used for reproducibility of the random initial conditions

vect0: 4 random initial conditions for plotting multiple trajectories

vect1, vect2 define each 2 random initial conditions for the second case, when m0 > 1
"""
np.random.seed(42)
vect0 = [(0.5+0.5*np.random.random(), 2.0+0.5*np.random.random()),
         (-0.5-0.5*np.random.random(), -0.5-0.5*np.random.random()),
         (0.5+0.5*np.random.random(), -0.5*np.random.random()),
         (-0.5-0.5*np.random.random(), 0.5*np.random.random())]
vect1 = [(0.5+0.5*np.random.random(), 2.0+0.5*np.random.random()),
         (-0.5-0.5*np.random.random(), -0.5-0.5*np.random.random())]
vect2 = [(0.5*np.random.random(), 0.5+0.5*np.random.random()),
         (-0.5*np.random.random(), 1.0+0.5*np.random.random())]



t = np.linspace(0, 500, 10000)

"""
Colours of the trajectories and parameter values used in the simulations
"""
colors = ['#d62728', '#ff7f0e', '#9467bd', '#1f77b4', '#17becf']
a = 1
b = 0
d = 1
k = 2.3
k_s = 16
eps = 0.1
m_0 = 0.8

"""
Setting the plot configurations 

"spines" -> Create the more conventional appearance of the axis
"""
fig, ax = plt.subplots(figsize = (14, 10), dpi = 100)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
ax.set_xlim(-1, 20)
ax.set_ylim(-2, 2)
ax.text(ax.get_xlim()[1] * 0.97, -0.15, 's', fontsize=14, fontweight='bold')
ax.text(-0.15, ax.get_ylim()[1] * 0.97, 'z', fontsize=14, fontweight='bold')
ax.set_title('S-NOD Phase Plane - Above Critical Basal Sensitivity (μ0 = 1.2)', 
             fontsize = 16, fontweight = 'bold', pad = 10)
ax.grid(True, linestyle = '--', alpha = 0.7)

# -------------------------------------------------Solution of equations and plotting --------------------------------------------------

"""
This loop solves the system for all the different initial conditions and plots the 
trajectories using the "quiver" function of the matplotlib library, for creating trajectories 
with arrows

The quiver (X, Y, U, V) method takes the following arguments:

X, Y : The starting points of the arrows (here, the s and z coordinates respectively)
U, V : The direction of the arrows (here, the difference between two consecutive s and z coordinates respectively)
In order to calculate the direction of the arrows, we find the difference between two consecutive points of the solution

odeint function is obsolete and solve_ivp is generally preferrable 
(docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html)
"""
for i, v in enumerate(vect0):
    #sol = odeint(system, v, t, full_output=0)
    sol = solve_ivp(system, [0, 500], [v[0], v[1]], t_eval=t,
                    args = (a, b, d, eps, k, k_s, m_0), dense_output=True)

    t = sol.t
    y = sol.y
    
    ax.quiver(y[1, :-1], y[0, :-1], y[1, 1:]-y[1, :-1], y[0, 1:]-y[0, :-1], 
              scale_units='xy', angles='xy', scale=1, color = colors[i],
              alpha = 1, linewidth = 2.5, label = f'Trajectory {i+1}')   


"""
For a = d = 1, from the definition of the S-NOD system, z(t) is bounded in the interval [-1, 1]
"""
z = np.linspace(-0.9999, 0.9999, 8001)

"""
----------------------------------------------------BEGIN------------------------------------------------------------------------
Plot the nullclines 
"""
ax.plot(k_s*z**4, z, color = '#d3d3d3', linewidth = 2.5, alpha = 0.4, 
        label = 's-nullcline')
ax.plot(z_null(z, a, b, d, k, m_0), z, color = '#222222', linewidth = 2.5, alpha = 0.4, 
        label = 'z-nullcline')

if (b == 0): 

    """
    For b = 0, as explained in the report, the z-nullcline has two branches, where one corresponds to the line z = 0. 
    This branch will not be computed by the z_null() method
    """
    ax.hlines(0, -1, 3, color = '#222222', linewidth = 2.5, alpha = 0.6, zorder = 3)

"""
----------------------------------------------------END------------------------------------------------------------------------
"""

"""
----------------------------------------------------BEGIN------------------------------------------------------------------------
Plot the equilibria

fsolve(f, x0, args = ()) finds the roots of the function f, given an initial guess x0 and additional arguments args

The 'root' variable contains a pair roots (corresponding to the two unstable equilibria of the system for m0 > 1) found by fsolve
We try 10 initial guesses on the entire range of the z values
root[2] == 1 indicates that the solution has converged successfully
Normally, for m0 < 1, no roots should have been found in this loop, however we still do a safety check
From m0 > 1, two roots should be found, corresponding to the two unstable equilibria of the system
"""
roots_found = []
for z_init in np.linspace(-0.99, 0.99, 10):
    root = fsolve(nullcline_intersection, z_init, args=(a, b, d, k, k_s, m_0), full_output=True)
    if root[2] == 1:
        z_val = root[0][0]
        print(np.abs(nullcline_intersection(z_val, a, b, d, k, k_s, m_0)), z_val)

        # We check that the root found is indeed an equilibrium (safety check)
        if np.abs(nullcline_intersection(z_val, a, b, d, k, k_s, m_0)) < 1e-6:
                if not any(np.abs(z_val - r) < 1e-4 for r in roots_found):
                    roots_found.append(z_val)

roots_found = np.array(roots_found)
ax.plot(k_s*roots_found**4, roots_found, marker = 'o', markersize = 10, markeredgecolor = 'black', markeredgewidth = 2,
        markerfacecolor = 'none', linestyle = 'none', label = 'Unstable Equilibrium')

"""
The 'eq' variable has the equilibrium near the origin
"""
eq = fsolve(equilibirum, 0, args = (a, b, d, k, m_0, k_s))
ax.plot(k_s*eq**4, eq, 'ko', markersize = 10, markeredgecolor = 'black', markeredgewidth = 2, 
        label = 'Equilibrium')

"""
----------------------------------------------------END------------------------------------------------------------------------
"""

"""
Create legend and show the plot
"""

ax.legend(loc = 'upper left', fontsize = 12, framealpha = 0.95)
plt.tight_layout()
plt.show()
   
