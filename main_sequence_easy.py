import numpy as np, matplotlib.pyplot as plt, time, math, sys
import multiprocessing as multiproc
from Main_File_Star1 import *
import time


L_sun = 3.846e26
R_sun = 6.955e8
M_sun = 1.989e30
sigma = 5.670373e-8      # Stefan-Boltzmann constant
np.seterr(all='ignore')

def make_star(t_central, star_number):
    print "*** Making star number: ", star_number, " , T_c =", t_central, "***"
    ms_star = Star(1e5, t_central).final_star
    print "a", ms_star.a, "central temperature", t_central, "surface temperature", ms_star.temp[ms_star.a]
    print "Surface Temperature", ms_star.temp[ms_star.a]
    print "Recalculated Temperature", (ms_star.luminosity[ms_star.a]/(4.0 * np.pi * sigma * ms_star.radius[ms_star.a]**2.0))**(1.0/4.0)
    print "Luminosity", ms_star.luminosity[ms_star.a]
    print "Luminosity/L_sun", ms_star.luminosity[ms_star.a]/L_sun
    print "Temperature^4", 4.0 * np.pi * sigma * ms_star.radius[ms_star.a]**2.0 * ms_star.temp[ms_star.a]**4.0
    return ms_star

###### Looping for a bunch of stars ############

surface_temperatures = []
stars_luminosities = []
stars_masses = []
stars_radii = []
stars = []
star_number = 0
start_time = time.time()

#central_temperatures = [8.23e6]
central_temperatures = 10.**np.arange(6.6, 7.5, 0.03)
print "Central temperatures: ", central_temperatures

for t_central in central_temperatures:
    star_number += 1
    ms_star = make_star(t_central, star_number)
    stars.append(ms_star)
    surface_temperatures.append((ms_star.luminosity[ms_star.a]/(4.0 * np.pi * sigma * ms_star.radius[ms_star.a]**2.0))**(1.0/4.0))
    stars_luminosities.append(ms_star.luminosity[ms_star.a] / L_sun)
    stars_masses.append(ms_star.mass[ms_star.a] / M_sun)
    stars_radii.append(ms_star.radius[ms_star.a] / R_sun)

print "done"
print "total time", time.time() - start_time
print "Surface temperatures:", surface_temperatures
print "Luminosities:", stars_luminosities
print "Masses:", stars_masses
print "Radii:", stars_radii


plt.figure(1)
plt.plot(surface_temperatures, stars_luminosities)
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()
plt.xlabel('Surface Temperatures (K)')
plt.ylabel('Luminosity ($L/L_{sun}$)')
plt.title('Main Sequence: Hertzsprung-Russell Diagram')
plt.savefig("main_sequence.png")

plt.figure(2)
plt.plot(stars_masses, stars_luminosities, '.')
#plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()
plt.xlabel('Masses ($M/M_{sun}$)')
plt.ylabel('Luminosity ($L/L_{sun}$)')
plt.title('Main Sequence: Luminosity versus Mass')
plt.savefig("main_sequence_luminosities_masses.png")

plt.figure(3)
plt.plot(stars_masses, stars_radii, '.')
#plt.xscale('log')
#plt.yscale('log')
plt.gca().invert_xaxis()
plt.xlabel('Masses ($M/M_{sun}$)')
plt.ylabel('Radii ($R/R_{sun}$)')
plt.title('Main Sequence: Radius versus Mass')
plt.savefig("main_sequence_radii_masses.png")

plt.show()
