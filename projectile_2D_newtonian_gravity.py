import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci

##CONSTANTS

#Planet - earth
G = 6.6742*10**-11
Rplanet = 6357000.0 #meters
mplanet = 5.972e24 #kg

#Rocket
mass = 640.0 / 1000.0 #kg

#Gravitational Acceleration Model
def gravity(x, z):
    global Rplanet, mplanet

    #Compute distance from the center of the planet
    r = np.sqrt(x**2 + z**2) 

    if r < Rplanet:
        accelx = 0
        accelz = 0
    else:
        accelx = G*mplanet/(r**3)*x
        accelz = G*mplanet/(r**3)*z

    return np.asarray([accelx, accelz])

    

#Equations of motion
# F = ma = m*zddot
# z is the altitiude from the center of the planet along the north pole
# x is the altitude from the center along the equator
# zdot is velocity along z
# zddot is acceleration along z

def Derivatives(state, t):
    #Globals
    global mass

    #State vector
    x = state[0]
    z = state[1]
    velx = state[2]
    velz = state[3]

    #Compute zdot (kinematic relationship)
    zdot = velz
    xdot = velx

    #Compute the total forces
    gF = -gravity(x, z) * mass

    #Aerodynamics
    aeroF = np.asarray([0.0, 0.0])

    #Thrust
    thrustF = np.asarray([0.0, 0.0])

    Forces = gF + aeroF + thrustF

    #Compute acceleration
    ddot = Forces/mass

    #Compute the statedot
    statedot = np.asarray([xdot,zdot,ddot[0],ddot[1]])

    return statedot

##Main script

#Test surface gravity
print('Surface Gravity (m/s/^2)= ', gravity(0,Rplanet))

##Initial conditions
x0 = Rplanet #m
z0 = 0
r0 = np.sqrt(x0**2 + z0**2)
velz0 = np.sqrt(G*mplanet/r0)*1.1
velx0 = 100.0
stateinitial = np.array([x0, z0, velx0, velz0])

#Time window
period = 2*np.pi/np.sqrt(G*mplanet)*r0**(3.0/2.0)*1.5
tout = np.linspace(0,period,1000) #rocket in air for calculated period

#Numerical integration call
stateout = sci.odeint(Derivatives, stateinitial, tout)

#Rename variables
xout = stateout[:,0]
zout = stateout[:,1]
altitude = np.sqrt(xout**2+zout**2) - Rplanet
velxout = stateout[:,2]
velzout = stateout[:,3]
velout = np.sqrt(velxout**2 + velzout**2)

###Plot

# Altitude
plt.subplot(1, 2, 1)
plt.plot(tout, altitude)
plt.xlabel('Time (sec)')
plt.ylabel('Altitude (m)')
plt.grid()

# Velocity
plt.subplot(1, 2, 2)
plt.plot(tout, velzout)
plt.xlabel('Time (sec)')
plt.ylabel('Normal Speed (m/s)')
plt.grid()

# Show both plots
plt.tight_layout()

##2D Orbit
plt.figure()
plt.plot(xout,zout,'r-',label='Orbit')
plt.plot(xout[0],zout[0],'g*')
theta = np.linspace(0,2*np.pi,100)
xplanet = Rplanet*np.sin(theta)
yplanet = Rplanet*np.cos(theta)
plt.plot(xplanet,yplanet,'b-',label='Planet')
plt.grid()
plt.legend()
plt.show()