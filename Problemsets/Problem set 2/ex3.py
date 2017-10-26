import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import emcee
import opt

def load(fname):
	return pickle.load(open(fname, 'rb'))

def model(theta, x):
	a, b = theta
	return b * x + a
		
def lnL(theta, x, y, y_err):
	m = model(theta, x)
	return -0.5*np.sum(((y-m)/y_err)**2)
	
def wrapper_lnL(a, b, x, y, y_err):
	return lnL((a,b), x, y, y_err)
	
def lnprior(theta):
	a, b = theta
	if -5.0 < a < 5.0 and -10.0 < b < 10.0:
		return 0.
	return -np.inf
	
def lnprob(theta, x, y, y_err):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnL(theta, x, y, y_err)
	

d = load('points_example1.pkl')
x = d['x']
y = d['y']
y_err = d['sigma']

estimate1 = opt.find_maxima(wrapper_lnL, [-5., -10.], [5., 10.], f_args = [x, y, y_err])[0]

print(lnL(estimate1, x, y, y_err))

ndim, nwalkers = 2, 100
pos = [p_init + 1E-4*np.random.randn



