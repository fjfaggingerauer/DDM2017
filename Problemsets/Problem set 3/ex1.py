import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table 
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold
from scipy import interpolate

def cv1(x, bws, model='gaussian', n_folds=10):
    """
    This calculates the leave-one-out cross validation. If you set 
    plot to True, then it will show a big grid of the test and training
    samples with the KDE chosen at each step. You might need to modify the 
    code if you want a nicer layout :)
    """
    cv_1 = np.zeros(len(bws))

    # Loop over each band-width and calculate the probability of the 
    # test set for this band-width
    for i, bw in enumerate(bws):
    
        # I will do N-fold CV here. This divides X into N_folds
        kf = KFold(n_folds)
        lnP = 0.0
                                 
        # Loop over each fold
        for train, test in kf.split(x):
            x_train = x[train, :]
            x_test = x[test, :]
            
            # Create the kernel density model for this bandwidth and fit
            # to the training set.
            kde = KernelDensity(kernel=model, bandwidth=bw).fit(x_train)
                                 
            # score evaluates the log likelihood of a dataset given the fitted KDE.
            lnP += kde.score(x_test)
            
        # Calculate the average likelihood          
        cv_1[i] = lnP/len(x)
        
    return cv_1
    
def get_best_kde(x, bws, model, n_folds):
	db = 2.*np.abs(bws - np.roll(bws, 1)).max()
	cv = cv1(x, bws, model, n_folds = n_folds)
	bw_max = bws[np.argmax(cv)]
	bw_big = np.linspace(max(bw_max-db, bws.min()),min(bw_max+db, bws.max()), 1000)
	f = interpolate.interp1d(bws, cv, kind=3)
	cv_big = f(bw_big)
	bw_opt = bw_big[np.argmax(cv_big)]
	
	return bw_opt, KernelDensity(kernel = model, bandwidth = bw_opt).fit(x)

if __name__ == '__main__':

	t = Table().read('joint-bh-mass-table.csv')
	n_folds = 20
	bws = np.linspace(1., 5., 100)
	bw_big = np.linspace(3.,5., 1000)

	data = np.array(t['MBH']).reshape(-1,1)
	models = ['gaussian']#,  'epanechnikov', 'exponential', 'linear', 'cosine','tophat',]

	for model in models:
		cv = cv1(data, bws, model, n_folds = n_folds)
		bw_max = bws[np.argmax(cv)]
		print(bw_max)
		bw_big = np.linspace(max(bw_max-0.5,bws.min()),bw_max+0.5, 1000)

		f = interpolate.interp1d(bws, cv, kind=3)
		cv_big = f(bw_big)
		print "The cubed interpolated maximum for {0} is {1:.4g}".format(model, bw_big[np.argmax(cv_big)])


		plt.plot(bws, cv, label = model)
		#plt.plot(bw_big, cv_big)
	plt.xlabel('bandwidth')
	plt.ylabel('CV likelihood')
	#plt.text(0.4, 0.04, 'Best BW={0:.4f}'.format(bws[np.argmax(cv)]))
	plt.legend(loc = 'lower right')

	plt.show()

	#print(t)
