"""
Made by Rhys Thayer

This program numerically solves the motion of a damped pendulum using Taylor series expansion to approximate the solution to the differential equation governing its motion.
I tested the accuracy of this code with the trusty eyeball method  comparing it to a rock I tied to a piece of dental floss. The decay is pretty close.
"""

import math
import matplotlib.pyplot


g = 9.81
l = 1
delta_t = 1/1000
theta = math.pi/2    #Whatever you put here is theta naught
omega = 0    #Whatever you put here is the initial angular frequency
drag_coefficient = 0.5 #close enough for a sphere or a rock
radius = 0.02 #meters
front_area = math.pi*(radius**2)
pressure = 102000 #pascalls
temperature = 301 #Kelvin
R_gas_constant = 8.31446261815324
molar_mass_air = 0.0289652
air_density = (pressure/(temperature*(R_gas_constant/molar_mass_air)))
mass = 0.01
drag_constants = (drag_coefficient*front_area*air_density*(l**2))/mass

t_values = []
theta_values = []


def dy_dx(theta):
    return ((-g/l)*math.sin(theta)) - drag_constants*omega

def d2y_dx2(theta):
    return ((-g/l)*math.cos(theta))

def d3y_dx3(theta):
    return ((g/l)*math.sin(theta))



for n in range(50000):
    t_values.append(n*delta_t)
    theta_values.append(theta)
    theta += delta_t * omega  # omega is rad/s and multiplied by delta_t equals radians, for theta
    omega += delta_t*(dy_dx(theta)) + ((delta_t**2)/2)*(d2y_dx2(theta)) + ((delta_t**3)/6)*(d3y_dx3(theta)) # apply the differential equation to omega because I substituted the derivative of omega as the second derivative of theta 

# print data points
for i in range(len(t_values)):
    if i%50 == 0:
        print(f"(t={t_values[i]:.3f}, θ={theta_values[i]:.3f})")

matplotlib.pyplot.plot(t_values, theta_values)
matplotlib.pyplot.xlabel("time (s)")
matplotlib.pyplot.ylabel("theta (radians)")
matplotlib.pyplot.isinteractive()
matplotlib.pyplot.show()