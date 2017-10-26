import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import scipy.ndimage as im

def show(a):
    plt.figure()
    if (len(a.shape) == 2):
        plt.imshow(a, interpolation = 'nearest', cmap = 'viridis')
    elif (len(a.shape) == 1):
        plt.plot(a)
    plt.show()


def find_maxima(f,
                lim1,
                lim2,
                samplings = 1,
                iters = 1000,
                convolve_size = 5.,
                hill_climb_factor = 1000.,
                f_args = None):
					
    def fp(f, x, f_args):
        if not isinstance(f_args, type(None)):
            return f(*(list(x)+f_args))
        else:
            return f(*x)
	
    def get_coords_from_index(index, coords):
        index = map(int, index)
        return [coords[i][j] for i,j in enumerate(index)]
	
    def sampling_max_find(f, lim1, lim2, iterations, convolve_size = 5., f_args = None):
        coords = [np.linspace(*i) for i in zip(lim1, lim2, iterations)]
        dims = len(lim1)
        ball_c = [np.linspace(-1.,1., convolve_size) for d in range(dims)]
        ball = ((np.array(np.meshgrid(*ball_c, indexing = 'ij'))**2).sum(axis=0) <= 1.).astype(float)
        
        res = np.zeros(iterations).flatten()
        for i, x in enumerate(product(*coords)):
			res[i] = fp(f, x, f_args)
        res = res.reshape(iterations)
        
        res_maxima = np.ones(iterations)
        for d in range(dims):
            sign_diff = np.sign(np.roll(res, -1, axis = d)-np.roll(res, 1, axis = d))
            maxima = ((sign_diff-np.roll(sign_diff, 1, axis = d)) == -2).astype(float)
            maxima = (im.convolve(maxima, ball, mode = 'constant', cval = 0.0) > 0)
            res_maxima *= maxima

        lbl, lbl_num = im.label(res_maxima)
        maxima_coords = im.measurements.center_of_mass(lbl, lbl, range(1,lbl_num+1))

        #print(maxima_coords)
        return tuple([get_coords_from_index(a, coords) for a in maxima_coords])

    def hill_climb(f, x0, eps_begin, iters = 100, eps_end = None, return_trail = False, f_args = None):
        def e(i, dims): # create basis vector in direction i
            a = np.zeros(dims)
            a[i] = 1.
            return a
        
        x = x0.copy()
        if eps_end == None: # set eps_end if not given
            eps_end = eps_begin/10.

        lowering_param = (eps_begin/eps_end)**(1./iters)
        eps = eps_begin
        dims = len(x)
        if return_trail: #create trail if necessary
            trail = np.zeros((dims, iters))
        for i in range(iters):
            ds = 0.
            dx = np.zeros(dims)
            for d in range(dims):
                x1 = x+eps*e(d, dims) # move a bit one way in direction d
                x2 = x-eps*e(d, dims) # move the other way a bit 
                dx[d] = fp(f,x1,f_args) - fp(f,x2,f_args) # check which way is best
                  
            ds = np.sqrt((dx**2).sum()) # normalize distance
            if ds == 0: # if f(x1)==f(x2) for all d we quit to avoid dividing by 0
                if return_trail:
                    return trail[:, :i]
                else:
                    return x

            x += eps*dx/ds # change our current estime to a better one
            eps /= lowering_param # lower eps so we move less next iteration
            if return_trail:
                trail[:,i] = x
        if return_trail:
            return trail
        else:
            return x


	if not isinstance(f_args, type(None)):
		f_args = list(f_args)
    if type(samplings) == int:
        samplings = [100 for i in lim1]
    approx_maxima = sampling_max_find(f, lim1, lim2, samplings, convolve_size, f_args)
    step_size = max((l2-l1)/it for l1, l2, it in zip(lim1, lim2, samplings))
    return np.array([hill_climb(f, np.array(m), step_size, iters, step_size/hill_climb_factor, False, f_args) for m in approx_maxima])

    




def f(x, a):
    return (x.sum(axis=0))*np.exp(-(x**2).sum(axis=0)/x.shape[0])

def g(x):
    return np.sin(x.sum(axis=0))/x.sum(axis=0)

if __name__ == '__main__':
    xlim1 = [-2.]
    xlim2 = [2.]

    x = np.array([np.linspace(xlim1[0], xlim2[0], 10000)])
    fx = f(x)
    plt.plot(x[0],fx)

    maxima = find_maxima(f, xlim1, xlim2, f_args = [1.])
    #print(maxima/np.pi)

    print(maxima.T[0], f(maxima.T))
    plt.scatter(maxima.T[0], f(maxima.T))
    plt.show()



