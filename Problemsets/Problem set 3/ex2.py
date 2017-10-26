import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from ex1 import get_best_kde 

def get_chance_from_cdf(m2, m1, cdf, tolerance = 0.02):
	if not np.isfinite(m2):
		m2 = None
	if not np.isfinite(m1):
		m1 = None
	try:
		arg1 = (np.abs(ms - m1) < tolerance)
		p1 = cdf[arg1].mean()
	except:
		p1 = 1.
		
	try:
		arg2 = (np.abs(ms - m2) < tolerance)
		p2 = cdf[arg2].mean()
	except:
		p2 = 0.
	return p1-p2


t = Table().read('pulsar_masses.vot')

masses = np.array(t['Mass'])
bws = np.linspace(0.01, 1., 100)
bw_opt, kde = get_best_kde(masses[:,None], bws, 'gaussian', 10) 
print(bw_opt)

ms = np.linspace(-20., 20., 100000)
pdf = np.exp(kde.score_samples(ms[:, None]))

cdf = np.cumsum(pdf)
cdf /= cdf[len(cdf)-1]

p1 = get_chance_from_cdf(1.8, np.inf, cdf)
p2 = get_chance_from_cdf(1.36, 2.26, cdf)
p3 = get_chance_from_cdf(0.86, 1.36, cdf)
print("Chance estimate to find mass > 1.8 MSun: {0:.4g}".format(p1))
print("Chance for a mass to lie in the range [1.36, 2.26]: {0:.4g}".format(p2))
print("Chance for a mass to lie in the range [0.86, 1.36]: {0:.4g}".format(p3))
print("Chance for binary with these mass ranges: {0:.4g}".format(p2*p3))

print("Avg neutron star mass: {0:.4g}".format(kde.sample(1000).mean()))


plt.plot(ms, pdf)
plt.plot(ms, cdf)
plt.scatter(masses, np.zeros(masses.shape))
plt.axis(xmin=0.5, xmax = 3.)
plt.show()




