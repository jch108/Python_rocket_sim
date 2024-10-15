import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci

##CONSTANTS
#Rocket
mass = 640.0 / 1000.0 #kg

#Equations of motion
# F = ma = m*zddot
# z is the altitiude, positive up
# zdot is velocity
# zddot is acceleration

def Derivatives(state, t):
    #Globals
    global mass

    #State vector
    z = state[0]
    velz = state[1]

    #Compute zdot (kinematic relationship)
    zdot = velz

    #Compute the total forces
    g = -9.81 * mass

    #Aerodynamics
    aero = 0.0

    #Thrust
    thrust = 0

    Forces = g + aero + thrust

    #Compute acceleration
    zddot = Forces/mass

    #Compute the statedot
    statedot = np.array([zdot, zddot])

    return statedot

##Main script

#Initial conditions
z0 = 0.0 #m
velz0 = 164.0 #m/s
stateinitial = np.array([z0, velz0])

#Time window
tout = np.linspace(0,30,1000) #rocket in air for 30s

#Numerical integration call
stateout = sci.odeint(Derivatives, stateinitial, tout)

#Rename variables
zout = stateout[:,0]
velzout = stateout[:,1]

###Plot

# Altitude
plt.subplot(1, 2, 1)
plt.plot(tout, zout)
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
plt.show()