'''
This code computes the background fluid recirculation in the capillary of a 
Spinning Drop Tensiometer, based on physical parameters

'''

import numpy as np
import matplotlib.pyplot as plt
import timeit
from sympy import init_printing
init_printing(use_latex=True)
from matplotlib import pyplot, cm


def pressure_Poisson(p, u, v, dx, dy, rho, dt):
    pn = np.empty_like(p)
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                      (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2)))
        
        # BC are renewed at each iteration
        # BC on pressure
        
        # Dirichlet-Neumann BC -> Walls, closed domain
        p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[-1, :] = p[-2, :] # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0

    return p
    
def parabola(x):
    a = x[0]
    b = x[-1]
    y = (((a-b)/2)**2 - (x -(a+b)/2)**2)*4/(a-b)**2
    return y
    
def half_parabola_left(x):
    a = x[0]
    b = x[-1]
    y = (((a-b))**2 - (x -(a))**2)*(a-b)**2
    return y

def half_parabola_right(x):
    a = x[0]
    b = x[-1]
    y = (((a-b))**2 - (x -(b))**2)*(a-b)**2
    return y

def spezzata_left(x):
    a = x[0]
    b = x[-1]
    y = -1/(b-a)*(x-a)+1
    return y


def spezzata_right(x):
    a = x[0]
    b = x[-1]
    y = 1/(b-a)*(x-a)
    return y

#########################################

start = timeit.default_timer()

#########################################
# Variable declaration
nx = 41
ny = 21
dx = 4 / float((nx - 1))
dy = 2 / float((ny - 1))
x = np.linspace(0, 4, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

nt = 500 # Apparent-time iterations
nit = 50 # Iteration for p at each solution step

# Physical variables
rho = 1
deltaRho = 0.05
nu = 2
dt = .001
omega = 0.3
r = 1.5

# Forcing

a = int((8*nx/20))
b = int((9*nx/20))
c = int((11*nx/20))
d = int((12*nx/20))
Xa = x[a:b]
Xb = x[c:d]

P = np.ones((ny, nx))
P[:, a:b] = spezzata_left(Xa)
P[:, b:c] = np.zeros(np.shape(P[:, b:c]))
P[:, c:d] = spezzata_right(Xb)

P[:,:] = P[:,:] *  deltaRho/rho*omega**2*r**2

for i in range(len(y)):
    P[i, :] = P[i, :] +rho*omega**2*y[i]**2 + rho*omega**2*r**2

#########################################
# Initial conditions
u = np.zeros((ny, nx))
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

#########################################
# Problem solution (2D)
# Solve for p in every iteration and use p to get u, v

for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    p = pressure_Poisson(p, u, v, dx, dy, rho, dt) # Iterate for p

    # Solve for u, with forcing directly on P (pressure constant term)
    # instead of dP/dx
    u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                    un[1:-1, 1:-1] * dt / dx *
                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                    vn[1:-1, 1:-1] * dt / dy *
                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                    dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                    nu * (dt / dx**2 *
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) -
                    dt / (2 * rho * dx) * (P[1:-1, 2:] - P[1:-1, 0:-2]))
    
    v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                    un[1:-1, 1:-1] * dt / dx *
                    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                    vn[1:-1, 1:-1] * dt / dy *
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                    dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                    nu * (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])) -
                    dt / (2 * rho * dy) * (P[2:, 1:-1] - P[0:-2, 1:-1]))
    
    # BC are renewed at each iteration
    # BC on velocity
    # Wall BC for closed domain (Dirichlet)
    u[0, :]  = 0
    u[:, 0]  = 0
    u[:, -1] = 0
    u[-1, :] = 0    
    v[0, :]  = 0
    v[-1, :] = 0
    v[:, 0]  = 0
    v[:, -1] = 0

#########################################
# Plot the results

levels = 20
# Velocity field
fig = pyplot.figure(figsize=(11,7), dpi=100)
plt.axis('equal')
# plotting the pressure field as a contour
plt.contourf(X, Y, p, np.linspace(
        np.min(np.asarray(p)), np.max(np.asarray(p)), levels),
        alpha=0.5, cmap=cm.viridis)  
plt.colorbar()
# plotting velocity field
plt.quiver(X[::1, ::1], Y[::1, ::1], u[::1, ::1], v[::1, ::1]) 
plt.xlabel('X')
plt.ylabel('Y')

# Streamlines
fig = pyplot.figure(figsize=(11, 7), dpi=100)
plt.axis('equal')
plt.contourf(X, Y, p, np.linspace(
        np.min(np.asarray(p)), np.max(np.asarray(p)), levels),
        alpha=0.5, cmap=cm.viridis) 
# Same as before for pressure
plt.colorbar()
plt.streamplot(X, Y, u, v)
plt.xlabel('X')
plt.ylabel('Y')

#########################################

np.savetxt('Poiseille_omega0.3_deltaRho0.05_v-4_8-9-11-12_r1.5.txt', 
           v[-4, a:d] - v[-4, a], delimiter = ' ', newline = '\n\r')

#########################################


plt.show()

stop = timeit.default_timer()
total_time = stop - start
print('Total execution time: '+ str(total_time)+' s')