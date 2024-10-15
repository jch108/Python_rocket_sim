import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci

plt.close("all")

##CONSTANTS

#Planet - smaller earth (easier to deal with)
G = 6.6742*10**-11
Rplanet = 600000  # meters
mplanet = 5.2915158 * 10**22  # kg

## PARAMETERS OF ROCKET
### Initial Conditions for single stage rocket
x0 = Rplanet
z0 = 0.0
velz0 = 0.0
velx0 = 0.0
period = 500.0
weighttons = 5.3
mass0 = weighttons * 2000 / 2.2  # kg
max_thrust = 167970.0
Isp = 250.0  # specific impulse - seconds
tMECO = 43.0  # main engine cutoff time
tsep1 = 2.0  # length of time to remove 1st stage
mass1tons = 1.0
mass1 = mass1tons * 2000 / 2.2

## Gravitational Acceleration Model
def gravity(x, z):
    global Rplanet, mplanet

    r = np.sqrt(x**2 + z**2)

    if r < Rplanet:
        accelx = 0.0
        accelz = 0.0
    else:
        accelx = G * mplanet / (r**3) * x
        accelz = G * mplanet / (r**3) * z

    return np.asarray([accelx, accelz])

def propulsion(t, theta):
    global max_thrust, Isp, tMECO, ve
    ## Timing for thrusters
    if t < tMECO:
        # We are firing the main thruster
        thrustF = max_thrust
        mdot = -thrustF / ve
    elif t > tMECO and t < (tMECO + tsep1):
        # Rocket is in process of dropping first stage
        thrustF = 0.0
        mdot = -mass1 / tsep1
    else:
        # After separation - no thrust or mass change
        thrustF = 0.0
        mdot = 0.0

    ## Angle of thrust
    thrustx = thrustF * np.cos(theta)
    thrustz = thrustF * np.sin(theta)

    return np.asarray([thrustx, thrustz]), mdot

### Equations of Motion
def Derivatives(state, t, theta):
    # state vector
    x = state[0]
    z = state[1]
    velx = state[2]
    velz = state[3]
    mass = state[4]

    # Compute zdot - Kinematic Relationship
    zdot = velz
    xdot = velx

    ### Compute the Total Forces
    # Gravity
    gravityF = -gravity(x, z) * mass

    # Aerodynamics
    aeroF = np.asarray([0.0, 0.0])

    # Thrust
    thrustF, mdot = propulsion(t, theta)

    Forces = gravityF + aeroF + thrustF

    # Compute Acceleration
    if mass > 0:
        ddot = Forces / mass
    else:
        ddot = 0.0
        mdot = 0.0

    # Compute the statedot
    statedot = np.asarray([xdot, zdot, ddot[0], ddot[1], mdot])

    return statedot

########### MAIN SCRIPT ###

# Test Surface Gravity
print('Surface Gravity (m/s^2) = ', gravity(0, Rplanet))

# Compute Exit Velocity
ve = Isp * 9.81  # m/s

# Time window
tout = np.linspace(0, period, 1000)

# Angles to test
angles = [10 * np.pi / 180.0, 70 * np.pi / 180.0]

# Create subplots for results
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle('Single Stage Rocket Simulation for Different Launch Angles')

for i, theta in enumerate(angles):
    # Populate Initial Condition Vector
    stateinitial = np.asarray([x0, z0, velx0, velz0, mass0])

    ### Numerical Integration Call
    stateout = sci.odeint(Derivatives, stateinitial, tout, args=(theta,))

    # Rename variables
    xout = stateout[:, 0]
    zout = stateout[:, 1]
    altitude = np.sqrt(xout**2 + zout**2) - Rplanet
    velxout = stateout[:, 2]
    velzout = stateout[:, 3]
    velout = np.sqrt(velxout**2 + velzout**2)
    massout = stateout[:, 4]

    # Plot altitude
    axs[0, i].plot(tout, altitude)
    axs[0, i].set_title(f'Altitude vs Time (Theta = {theta * 180 / np.pi:.0f} degrees)')
    axs[0, i].set_xlabel('Time (sec)')
    axs[0, i].set_ylabel('Altitude (m)')
    axs[0, i].grid()

    axs[0, 0].set_ylim(0, 350000)
    axs[0, 1].set_ylim(0, 350000)

    # Plot velocity
    axs[1, i].plot(tout, velout)
    axs[1, i].set_title(f'Velocity vs Time (Theta = {theta * 180 / np.pi:.0f} degrees)')
    axs[1, i].set_xlabel('Time (sec)')
    axs[1, i].set_ylabel('Total Speed (m/s)')
    axs[1, i].grid()

    axs[1, 0].set_ylim(0, 3000)
    axs[1, 1].set_ylim(0, 3000)

    # Plot mass
    axs[2, i].plot(tout, massout)
    axs[2, i].set_title(f'Mass vs Time (Theta = {theta * 180 / np.pi:.0f} degrees)')
    axs[2, i].set_xlabel('Time (sec)')
    axs[2, i].set_ylabel('Mass (kg)')
    axs[2, i].grid()

## Show all plots
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

## 2D Orbit plot
plt.figure()
for theta in angles:
    # Populate Initial Condition Vector
    stateinitial = np.asarray([x0, z0, velx0, velz0, mass0])
    stateout = sci.odeint(Derivatives, stateinitial, tout, args=(theta,))
    
    xout = stateout[:, 0]
    zout = stateout[:, 1]
    
    plt.plot(xout, zout, label=f'Theta = {theta * 180 / np.pi:.0f} degrees')

plt.plot(xout[0], zout[0], 'g*', label='Launch Point')
theta = np.linspace(0, 2 * np.pi, 1000)
xplanet = Rplanet * np.sin(theta)
yplanet = Rplanet * np.cos(theta)
plt.plot(xplanet, yplanet, 'b-', label='Planet')
plt.grid()
plt.xlabel('X Position (m)')
plt.ylabel('Z Position (m)')
plt.title('2D Orbit Paths for Different Launch Angles')
plt.legend()
plt.axis('equal')
plt.show()
