import numpy as np, matplotlib.pyplot as plt, time, math, sys



surface_temperatures = [1805.2,1868.6,1898,1963,2064,2158,2276,2424,2571,2746,2944,3170,3429,3723,4056,4433,4855,5331,5870,6465,7127,7764,8623,9923,11574,13968,16964,20547,24616,29334,35157]
stars_luminosities = [0.0009714,.001292,.001669,.002425,.0040397,.006135,.009233,.01388,.02086,.03134,.04714,.07087,.10657,.1602,.2410,.3624,.5451,.8120,1.231,1.846,2.751,4.103,6.807,14.04,36.50,109.31,346.2,1125,3753,13059,49692]
stars_masses = [.2098,.2353,.2696,.3228,.3943,.4435,.4849,.5238,.5636,.6044,.6472,.6928,.7414,.7935,.8493,.9091,.9732,1.042,1.116,1.195,1.280,1.375,1.513,1.743,2.117,2.713,3.679,5.272,7.860,12.31,21.05]
stars_radii = [.3195,.3438,.3790,.4271,.4986,.5621,.6193,.6697,.7297,.7843,.8371,.8852,.9272,.9646,.9964,1.024,1.046,1.064,1.076,1.086,1.091,1.123,1.172,1.271,1.507,1.790,2.160,2.654,3.377,4.436,6.0245]

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
plt.xscale('log')
plt.yscale('log')
#plt.gca().invert_xaxis()
plt.xlabel('Masses ($M/M_{sun}$)')
plt.ylabel('Luminosity ($L/L_{sun}$)')
plt.title('Main Sequence: Luminosity versus Mass')
plt.savefig("main_sequence_luminosities_masses.png")

plt.figure(3)
plt.plot(stars_masses, stars_radii, '.')
plt.xscale('log')
plt.yscale('log')
#plt.gca().invert_xaxis()
plt.xlabel('Masses ($M/M_{sun}$)')
plt.ylabel('Radii ($R/R_{sun}$)')
plt.title('Main Sequence: Radius versus Mass')
plt.savefig("main_sequence_radii_masses.png")

plt.show()