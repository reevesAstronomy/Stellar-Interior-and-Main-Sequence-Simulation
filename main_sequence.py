import numpy as np, matplotlib.pyplot as plt, time, math, sys
import multiprocessing as multiproc
from Test1 import Star
import time


L_sun = 3.846e26
sigma = 5.670373e-8      # Stefan-Boltzmann constant
np.seterr(all='ignore')

def make_star(t_central):
    print "making a star: T_c", t_central
    ms_star = Star(1e6, t_central, [1.0, 1.0, 1.0]).final_star
    print "a", ms_star.a, "central temperature", t_central, "surface temperature", ms_star.temp[ms_star.a]
    print "Surface Temperature", ms_star.temp[ms_star.a]
    print "Recalculated Temperature", (ms_star.luminosity[ms_star.a]/(4.0 * np.pi * sigma * ms_star.radius[ms_star.a]**2.0))**(1.0/4.0)
    print "Luminosity", ms_star.luminosity[ms_star.a]
    print "Luminosity/L_sun", ms_star.luminosity[ms_star.a]/L_sun
    print "Temperature^4", 4.0 * np.pi * sigma * ms_star.radius[ms_star.a]**2.0 * ms_star.temp[ms_star.a]**4.0
    return ms_star
        
###### Looping for a bunch of stars ############
def MainSequence():
    central_temperatures = []
    surface_temperatures = []
    stars_luminosities = []
    star_number = 0
    start_time = time.time()
    
    central_temperatures = map(lambda x: 10**x, np.arange(6.6, 7.5, 0.1))
    print central_temperatures


    # change the Pool argument to modify the number of processors
    p = multiproc.Pool(multiproc.cpu_count())
    ms_stars = p.map(make_star, central_temperatures)
    
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.gca().invert_xaxis()
    plt.ion()
    plt.show()
    surface_temperatures = map(lambda x: (x.luminosity[x.a]/(4.0 * np.pi * sigma * x.radius[x.a]**2.0))**(1.0/4.0), ms_stars)
    stars_luminosities = map(lambda x: x.luminosity[x.a], ms_stars)
    
    print "done"
    print "total time", time.time() - start_time

    plt.scatter(surface_temperatures, map(lambda x: x/L_sun, stars_luminosities))
    plt.savefig("main_sequence.png")
    plt.show()

MainSequence()
