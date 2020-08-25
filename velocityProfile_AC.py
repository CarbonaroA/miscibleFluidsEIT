'''
This code computes the velocity profile of the fluids in a coflow experiments
in a microfluidic channel, starting from the imposed flow rates and the 
interface position.

'''

import numpy as np
import matplotlib.pyplot as plt
import timeit
import mpmath as mp
from scipy.optimize import curve_fit


def quartica(x, A, B, C, X0):
    y = A*(x-X0)**4+B*(x-X0)**2+C
    return y

start = timeit.default_timer()

mp.dps = 1000

#########################################
# Definition of physical parameters
# z and y are the coordinates: arrays with nPoints
etaFactor = 5e4
eta1 = 0.049/etaFactor 
# (1 = 1 Pa*s, if changed watch out for Re) glycerol (or mixture) 
# eta1 = 0.001/etaFactor
eta2 = 0.001/etaFactor # H2O

rho1 = 1.26e3 # Kg/m3
rho2 = 1e3 # Kg/m3

S = eta1+eta2
D = eta1-eta2 

Q1 = 1/60*10e-9 # (m3/s) glycerol
Q2 = 1/60*400e-9 # (m3/s) H2O
Q_singFl = 1/60*400e-9 # (m3/s)

G = -2.63507E6/etaFactor 
# Given for the moment, to be calculated with interface position
v_ph = 0.02 # (m/s)

nPoints = 50
nPointsY = 50
nPointsZ = 50

b = 100.0e-6 # (m) Channel width -> Y
a = 1e-4 # (m) Channel height -> Z

YH2O = 35.0e-6 # (m) From H2O side, experimentally measured from there
Y = b - YH2O  # (m) Interface position (to be calculated) 
# FROM GLY SIDE, for the moment: measured on side of fluid 1 
# (invert eta1-2 to have it on H2O side)
                   
y = np.ndarray.tolist(np.linspace(0, b, nPointsY))
z = np.ndarray.tolist(np.linspace(0, a, nPointsZ))

M = 5 # Number of Fourier components in the profile

#########################################
# The velocity profile is to be calculated as a Fourier series
# Define the m-th element, then sum up to M
# Define the Fourier part of the term first: in Giovanni's notation, v/G.
# G is to be calculated later

u = np.zeros((nPointsZ, nPointsY))

for j in y:
    for k in z:
        u1 = 0
        u2 = 0
        
        for m in [x+1 for x in range(M)]:
            Km = 2*a**2*(1-(-1)**m)/(m**3*np.pi**3)
            Cm = S*mp.sinh(b*m*np.pi/a) + D*mp.sinh((b-2*Y)*m*np.pi/a)
            A1m = Km/eta1
            B1m = (Km/(eta1*Cm)) * (2*eta1 - S*mp.cosh(b*m*np.pi/a) + 
                  D*(mp.cosh((b-2*Y)*m*np.pi/a) - 2*mp.cosh((b-Y)*m*np.pi/a)) )
            A2m = (Km/(eta2*Cm)) * (2*(eta2 + 
                  D*mp.cosh(Y*m*np.pi/a))*mp.sinh(b*m*np.pi/a) - 
                  D*mp.sinh(2*Y*m*np.pi/a) )
            B2m = (Km/(eta2*Cm)) * (S - 2*mp.cosh(b*m*np.pi/a)*(eta2 + 
                  D*mp.cosh(Y*m*np.pi/a)) + D*mp.cosh(2*Y*m*np.pi/a) )
            
            u1 = u1 + np.sin(m*np.pi*k/a)* (A1m*mp.cosh(m*np.pi*j/a) + 
                             B1m*mp.sinh(m*np.pi*j/a))
            
            u2 = u2 + np.sin(m*np.pi*k/a)* (A2m*mp.cosh(m*np.pi*j/a) + 
                             B2m*mp.sinh(m*np.pi*j/a))
        
        u1 = u1 + (k*(k-a))/(2*eta1)
        u2 = u2 + (k*(k-a))/(2*eta2)
        
        #u1 = u1 + (k**2)/(2*eta1)
        #u2 = u2 + (k**2)/(2*eta2)
    
        if j <Y:
            u[z.index(k), y.index(j)] = u1
        else:
            u[z.index(k), y.index(j)] = u2

#########################################
# Calculate G

C1 = 0
C2 = 0

for iy in range(nPointsY):
    for iz in range(nPointsZ):    
        if y[iy]< Y:
            C1 = C1 + u[iz, iy]*y[1]*z[1]
        else:
            C2 = C2 + u[iz, iy]*y[1]*z[1]

G1 = Q1 / C1
G2 = Q2 / C2

print('G1 prime = ', str(G1*etaFactor), ' Pa/m')
print('G2 prime = ', str(G2*etaFactor), ' Pa/m')

G_m = (G1+G2)/2

u = u*G_m
#u = u*G2

'''
#Single fluid

C_singFl = 0
for iy in range(nPointsY):
    for iz in range(nPointsZ):    
        C_singFl = C_singFl + u[iz, iy]*y[1]*z[1]
        
G_singFl = Q_singFl/C_singFl
print('G single fluid = ', str(G_singFl*etaFactor), ' Pa/m')
u = u*G_singFl

print('Max v SF = ',  str(u.max()))

#Shear rate Sf _ water side
du = np.zeros((np.shape(u)))

for k in range(len(z)):
    for j in range(len(y)-1):
        du[k, j+1] = (u[k, j+1]-u[k, j])/y[1]

#yy, zz = np.meshgrid(y, z)
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(yy, zz, du)
#ax.set_xlabel('y')
#ax.set_ylabel('z')
#plt.title('du')

# u and so du oscillates slightly: 
# fit it (see below as for shear rate in multifluid)
poptSF, pcovSF = curve_fit(quartica, range(len(z)), du[:, 1]/max(du[:, 1]), 
                           [-1, -1, 1, nPointsZ//2], maxfev = 10000)
ASF = poptSF[0]
BSF = poptSF[1]
CSF = poptSF[2]
X0SF = poptSF[3]
fitRescaledDu = quartica(range(len(z)), ASF, BSF, CSF, X0SF)*max(du[:, 1])

plt.figure()
plt.plot(z, du[:, 1])
plt.plot(z, fitRescaledDu, 'c')

print('Max du SF = ', str(max(fitRescaledDu)), ' 1/s')

# Critical layer position (SF): distance from interface
#CritPosition = u[nPoints//2, :].index([y[i] for i in range(nPoints) if 
                                       u[nPoints//2, i] > v_ph][-1])
CritPosition = y[np.where(u[nPoints//2, :] > v_ph)[0][0]-1] 
# Given from below, to see when 0
print('Critical layer position = ', str(CritPosition*1e6), ' micron')
'''

#########################################

yy, zz = np.meshgrid(y, z)

fig = plt.figure()
ax = fig.gca(projection='3d')
#surf = ax.plot_surface(yy, zz, u, cstride=1, rstride=1)
surf = ax.plot_surface(yy, zz, u)
#surf = ax.plot_surface(yy, zz, zz**2)
#ax.set_aspect('equal')
#figsize=plt.figaspect(4)
#ax.auto_scale_xyz([0, b], [0, a], [0, u.max()])
ax.set_xlabel('y')
ax.set_ylabel('z')

#########################################
# Calculate shear rate jump, interface velocity, Re...

u_half = u[nPointsZ//2, :] # This is along y

# make a cross check on sufficiency of harmonics:
# take the coordinate of max(u_half), scan in the other direction
# Check that the middle position is really the max for u (monotony of u'')
du_half = np.zeros(len(u_half))

# Differentiate u_half(y)
for j in range(len(u_half)-1):
    du_half[j+1] = (u_half[j+1]-u_half[j])/y[1]

shearRateJump_half = du_half[y.index([i for i in y if i>Y][0])] - 
                             du_half[y.index([i for i in y if i<Y][-1])]
#print('Shear rate jump (half)', str(shearRateJump_half)) 
# Could iterate this on all z values to see dependence: below

du = np.zeros((np.shape(u)))
shearRateJump = np.zeros(len(z))

for k in range(len(z)):
    for j in range(len(y)-1):
        du[k, j+1] = (u[k, j+1]-u[k, j])/y[1]
    shearRateJump[k] = du[k, y.index([i for i in y if i>Y][0])] - 
                 du[k, y.index([i for i in y if i<Y][-1])]    

# dU on glycerol side of the interface
du_gly = du[:, y.index([i for i in y if i>Y][0])-1]
popt_gly, pcov_gly = curve_fit(quartica, range(len(du_gly)), du_gly/max(du_gly), 
                               [-1, -1, 1, nPointsZ//2], maxfev = 10000)
A_gly = popt_gly[0]
B_gly = popt_gly[1]
C_gly = popt_gly[2]
X0_gly = popt_gly[3]
fitRescaledDuGly = quartica(range(len(du_gly)), 
                            A_gly, B_gly, C_gly, X0_gly)*max(du_gly)
plt.figure()
plt.plot(z, du_gly)
plt.plot(z, fitRescaledDuGly, '.c')
plt.title('du gly side')
print('Max dU gly side = ', str(max(fitRescaledDuGly)), ' 1/s')

du_H2O = du[:, y.index([i for i in y if i>Y][0])]
popt_H2O, pcov_H2O = curve_fit(quartica, range(len(du_H2O)), 
                               du_H2O/max(du_H2O), [-1, -1, 1, nPointsZ//2], 
                                         maxfev = 10000)
A_H2O = popt_H2O[0]
B_H2O = popt_H2O[1]
C_H2O = popt_H2O[2]
X0_H2O = popt_H2O[3]
fitRescaledDuH2O = quartica(range(len(du_H2O)), 
                            A_H2O, B_H2O, C_H2O, X0_H2O)*max(du_H2O)
plt.figure()
plt.plot(z, du_H2O)
plt.plot(z, fitRescaledDuH2O, '.c')
plt.title('du H2O side')
print('Max dU H2O side = ', str(max(fitRescaledDuH2O)), ' 1/s')

#plt.figure()
#plt.plot(shearRateJump)

# The shear rate jump oscillates, depending on M: 
# fit it in some way? Not sufficient to take half of the channel
# For instance, with even powers (up to 4th)
rescaledShearRate = shearRateJump/max(shearRateJump) 
# Fit rescaling to unity, easier numerically
# Also fit not using z as axis, too small - numerically hard. 
# Fit against range(len(shearRate)), then rescale
popt2, pcov2 = curve_fit(quartica, range(len(rescaledShearRate)), 
                         rescaledShearRate, [-1, -1, 1, nPointsZ//2], 
                                            maxfev = 10000)
A2 = popt2[0]
B2 = popt2[1]
C2 = popt2[2]
X02 = popt2[3]
fitRescaledShear = quartica(range(len(rescaledShearRate)), 
                            A2, B2, C2, X02)*max(shearRateJump)
plt.figure()
plt.plot(z, shearRateJump)
plt.plot(z, fitRescaledShear)
plt.title('Shear rate jump')
#plt.plot(z, rescaledShearRate*max(shearRateJump), '.c')

print('Max shear rate jump : ', str(max(fitRescaledShear)), ' s^-1')

u_interface = np.mean([u[:,  y.index([i for i in y if i>Y][0])], 
                         u[:, y.index([i for i in y if i<Y][-1])]], axis = 0)
# u_interface oscillates as well, like shear rate jump: fit it?
rescaled_u_interf = u_interface/max(u_interface)
popt_u, pcov_u = curve_fit(quartica, range(len(rescaled_u_interf)), 
                           rescaled_u_interf, [-1, -1, 1, nPointsZ//2], 
                                              maxfev = 10000)
Au = popt_u[0]
Bu = popt_u[1]
Cu = popt_u[2]
X0u = popt_u[3]
fitRescaled_u_int = quartica(range(len(rescaled_u_interf)), 
                             Au, Bu, Cu, X0u)*max(u_interface)
plt.figure()
plt.plot(z, u_interface)
plt.plot(z, fitRescaled_u_int)
plt.title('Interface velocity')

#print('Max v interface (fit) = ', str(max(fitRescaled_u_int)))
#plt.plot(z, rescaled_u_interf*max(u_interface), '.c')

# define u_interface only on one side? gly?

u_interface_gly = u[:, y.index([i for i in y if i<Y][-1])]

rescaled_u_interf_gly = u_interface_gly/max(u_interface_gly)
popt_u_gly, pcov_u_gly = curve_fit(quartica, range(len(rescaled_u_interf_gly)), 
                                   rescaled_u_interf_gly, [-1, -1, 1, nPointsZ//2], 
                                              maxfev = 10000)
Au_gly = popt_u_gly[0]
Bu_gly = popt_u_gly[1]
Cu_gly = popt_u_gly[2]
X0u_gly = popt_u_gly[3]
fitRescaled_u_int_gly = quartica(
        range(len(rescaled_u_interf_gly)), 
             Au_gly, Bu_gly, Cu_gly, X0u_gly)*max(u_interface_gly)
plt.figure()
plt.plot(z, u_interface_gly)
plt.plot(z, fitRescaled_u_int_gly)
plt.title('Interface velocity glycerol')

print('Max v interface glycerol (fit) = ', str(max(fitRescaled_u_int_gly)))

# Re number: define it with interface velocity, and with max velocity 
# in the channel (so, in water generally)
# Re = rho*u*h/eta

# Re on u_interface (max)
rho = 1e3 # kg/m3
h = a
#eta = (eta1+eta2)/2*etaFactor # For the moment take the average
eta = eta2*etaFactor # Water
Re_interface = rho*h/eta*max(fitRescaled_u_int)
#print('Re defined at interface = ', str(Re_interface))

Re_Govindarajan = rho/eta*(Q1+Q2) 
# In Govindarajan2016, Q = tot vol flow rate PER UNIT DISTANCE
#print('Re Govindarajan = ', str(Re_Govindarajan))
Re_Govindarajan_h = rho/eta*(Q1+Q2)/h
#print('Re Govindarajan, corrected with h = ', str(Re_Govindarajan_h))
                          
# Re on max velocity in channel: 
# check for position vs Y to determine eta to take

y_max = np.argmax(u_half)
# Now move along z and fit to avoid oscillations
u_max = u[:, y_max]
#plt.figure()
#plt.plot(u_max) 
popt_uMax, pcov_uMax = curve_fit(quartica, range(len(u_max)), 
                                 u_max/max(u_max), [-1, -1, 1, nPointsZ//2], 
                                          maxfev = 10000)
AuMax = popt_uMax[0]
BuMax = popt_uMax[1]
CuMax = popt_uMax[2]
X0uMax = popt_uMax[3]
fitRescaled_u_Max = quartica(range(len(u_max)), 
                             AuMax, BuMax, CuMax, X0uMax)*max(u_max)
# Now check the position along y to determine which eta and rho to take
if y_max >= y.index([i for i in y if i>Y][0]):
    # This means fluid 2: for how it's written above, water
    eta_max = eta2
    rho_max = rho2
else:
    eta_max = eta1
    rho_max = rho1
eta_max = eta_max*etaFactor

print('Max v in the channel = ', str(max(fitRescaled_u_Max)))

h_max = YH2O # Take the distance interface-wall on max velocity side
Re_max = rho_max*h_max*max(fitRescaled_u_Max)/eta_max
print('Re defined with max velocity in the channel = ', str(Re_max))

Re_YH1 = rho1*max(fitRescaled_u_int_gly)*Y/eta1/etaFactor
Re_YH2 = rho2*max(fitRescaled_u_int_gly)*(b-Y)/eta2/etaFactor
print('Re Y&H1 = ', str(Re_YH1))                 
print('Re Y&H2 = ', str(Re_YH2))

# Critical layer position (coflow): absolute position (not 
# distance from interface)
# CritPosition = u[nPoints//2, :].index([y[i] for i in range(nPoints) 
# if u[nPoints//2, i] > v_ph][-1])
CritPosition = y[np.where(u[nPoints//2, :] > v_ph)[0][0]-1]
print('Critical layer position = ', str(CritPosition*1e6), ' micron') 
# From below (gly side), to see when == interface

#########################################

plt.show()

stop = timeit.default_timer()
total_time = stop - start
print('Total execution time: '+ str(total_time)+' s')