"""
Made by Rhys Thayer

This is a multi-body simulator that approximates the movement of particles under the influence of gravity and electric forces. 
It calculates the forces acting on each particle due to every other particle and updates their positions and velocities each time step.
"""

import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Particle:
    """
    This class makes and contains point particles that interact with each other gravitationally and electrically.
    Each particle contains a value for mass, charge, and 3D coordinates for position, velocity, and acceleration.

    """
    
    # Universal Constants
    GRAV_CON = 6.67430 * (10**-11)
    VAC_ELE_PERM = 8.8541878188 * (10**-12)
    COULOMB_CON = 1/(4*math.pi*VAC_ELE_PERM)
    SPEED_LIGHT = 299792458
    
    # Class Variables
    particles = []
    time = 0
    time_slice = 0.1

    def __init__(self, mass, charge, x_p, y_p, z_p, x_v = 0, y_v = 0, z_v = 0, x_a = 0, y_a = 0, z_a = 0):
        """
        Values for mass, charge, and 3D coordinates for position, velocity, and acceleration
        """
        self.mass = mass    #kg
        self.charge = charge    #Coulombs
        self.x_p = x_p    #meters
        self.y_p = y_p    #meters
        self.z_p = z_p    #meters
        self.x_v = x_v    #m/s
        self.y_v = y_v    #m/s
        self.z_v = z_v    #m/s
        self.x_a = x_a    #m/s^2
        self.y_a = y_a    #m/s^2
        self.z_a = z_a    #m/s^2

        Particle.particles.append(self)

    def __str__(self):
        particle_info = f"""
                Mass:   {self.mass} kg
                Charge: {self.charge} Coulombs
                x_pos:  {self.x_p} meters
                y_pos:  {self.y_p} meters
                z_pos:  {self.z_p} meters
                x_vel:  {self.x_v} m/s
                y_vel:  {self.y_v} m/s
                z_vel:  {self.z_v} m/s
                x_acc:  {self.x_a} m/s^2
                y_acc:  {self.y_a} m/s^2
                z_acc:  {self.z_a} m/s^2
                total distance is {math.sqrt((self.x_p)**2 + (self.y_p)**2 + (self.z_p)**2)} meters
                total velocity is {math.sqrt((self.x_v)**2 + (self.y_v)**2 + (self.z_v)**2)} m/s
                total acceleration is {math.sqrt((self.x_a)**2 + (self.y_a)**2 + (self.z_a)**2)} m/s^2
                """
        return particle_info
    
    def distance(the_particle, a_particle):
        """Finds the distance between two particles."""
        distance = ((the_particle.x_p - a_particle.x_p)**2 + (the_particle.y_p - a_particle.y_p)**2 + (the_particle.z_p - a_particle.z_p)**2)**(0.5)
        return distance

    def grav_force(the_particle, a_particle):
        """Finds the gravitational force between two particles and updates the acceleration of the first particle accordingly."""
        distance = Particle.distance(the_particle, a_particle)
        force = (Particle.GRAV_CON * ((the_particle.mass * a_particle.mass)/(distance**2)))

        the_particle.x_a += (force/the_particle.mass)*((a_particle.x_p - the_particle.x_p)/distance)
        the_particle.y_a += (force/the_particle.mass)*((a_particle.y_p - the_particle.y_p)/distance)
        the_particle.z_a += (force/the_particle.mass)*((a_particle.z_p - the_particle.z_p)/distance)

    def elec_force(the_particle, a_particle):
        """Finds the electrical force between two particles and updates the acceleration of the first particle accordingly."""
        distance = Particle.distance(the_particle, a_particle)
        force = (Particle.COULOMB_CON * ((the_particle.charge * a_particle.charge)/(distance**2)))

        the_particle.x_a += (force/the_particle.mass)*((the_particle.x_p - a_particle.x_p)/distance)
        the_particle.y_a += (force/the_particle.mass)*((the_particle.y_p - a_particle.y_p)/distance)
        the_particle.z_a += (force/the_particle.mass)*((the_particle.z_p - a_particle.z_p)/distance)

    
    def update_force(the_particle, particles):
        """Applies all forces on a particle from the class variable particles"""
        the_particle.x_a = 0
        the_particle.y_a = 0
        the_particle.z_a = 0

        for i in particles:
            if i != the_particle:
                the_particle.grav_force(i)
                the_particle.elec_force(i)

    @classmethod
    def move(cls):
        """
        Updates the position and velocity of all particles in the simulation based on the forces acting on them.
        
        :param cls: Particle class
        """
        delta_time = Particle.time_slice

        for i in Particle.particles:
            cls.update_force(i, Particle.particles)

            #Change velocity
            i.x_v += (i.x_a * delta_time)
            i.y_v += (i.y_a * delta_time)
            i.z_v += (i.z_a * delta_time)

            #Change Position
            i.x_p += (i.x_v * delta_time) + ((i.x_a * (delta_time**2))/2)
            i.y_p += (i.y_v * delta_time) + ((i.y_a * (delta_time**2))/2)
            i.z_p += (i.z_v * delta_time) + ((i.z_a * (delta_time**2))/2)


        if (((i.x_v)**2 + (i.y_v)**2 + (i.z_v)**2)**(0.5)) >= Particle.SPEED_LIGHT:
            print("\n!!!SPEED LIMIT REACHED!!!\n")

# Simulation parameters
Particle.time_slice = 0.1
seconds_run = 20
iterations_run = int(seconds_run / Particle.time_slice)
num_displays = 20

#Display parameters
#Set up for the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter([], [], [])

#where the window is zoomed
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

def get_particle_positions_at_frame():
    """Just as the name says, gets the positions of all particles at the current frame."""
    Particle.move()

    positions = []

    for i in Particle.particles:
        positions.append([i.x_p, i.y_p, i.z_p])

    return positions

def init():
    return scat,


def update(frame):
    positions = get_particle_positions_at_frame()
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    z = [p[2] for p in positions]
    scat._offsets3d = (x, y, z)
    return scat,

ani = FuncAnimation(fig, update, frames=num_displays,init_func=init, blit=False, interval=100)

#Example Particles
thing = Particle(0.01, 0.000001, 1, 0, 0, 0, 0.1)
point = Particle(0.01, -0.000001, -1, 0, 0, 0, -0.1)
spot = Particle(0.02, 0.000001, 0, 0, -2)
print(thing)
plt.show()