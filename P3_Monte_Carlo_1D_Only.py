import numpy as np
import math
from numba import njit, prange
import matplotlib.pyplot as plt
import time

'''
This module holds the ocde for one-dimensional Monte Carlo integration with importance sampling.
This runs faster than the general method so for weaker machines, I reccomomend this is used for higher accuracies.
The first time the code is run, Numba will compile it so this may take a little longer.
Therefore, I reccomend starting with low accuracies.

Result gathering code is at the bottom and can be changed for dfferent errors.
'''
#%%
@njit
def wavefunc(x):
    '''
    The ground state 1D SHO wavefunction.
    Returns the value of the function at x.
    
    Parameters
    ----------
    x : flt
        The x value you want to evaluate the function for.
    
    Returns
    -------
    f : flt
        The value of the function at x.
    '''
    return (1/math.sqrt(np.pi)) * math.exp(-x**2)

#%%

@njit
# Importance sampling, PDF and inverse CDF depend on the integration range
def pdf(a,b,x,importance):
    '''
    A function of linear or uniform PDFs that depend on limits of integration.
    Returns the value of a linear or uniform PDF at x.
    
    Parameters
    ----------
    a : flt
        The lower limit of integration.
    b : flt
        The upper limit of integration.
    x : flt
        The point being evaluated.
    importance : str
        The PDF desired..
        
    Raises
    ------
    TypeError : str
        If an invalid PDF is requested.
        
    Returns
    -------
    f : flt
        The value of the PDF at x.
    '''
    if importance == str('uniform'):
        if x < a or x > b:
            #if point is outside of integration limits return 0
            return 0
        else:
            #for uniform pdf return flat, normalised value
            return 1/(b-a)
    if importance == str('linear'):
        if x < a or x > b:
            #if point is outside of integration limits return 0
            return 0
        else:
            #PDF normalised irrespective of integration range
            A = -0.48 / ((b - a) * (-0.24 * (b + a) + 0.98))
            B = 0.98 / ((b - a) * (-0.24 * (b + a) + 0.98))
            return A * x + B
    else:
        #if PDF isn't linear or uniform return an error
        raise TypeError('Invalid PDF')
        
@njit(parallel=True)
def metropolis(func,a,b,mu,N,importance):
    '''
    Function to carry out MCMC simulation.
    Returns the integral estimate.
    
    Parameters
    ----------
    func : function
        The function to be integrated.
    a : flt
        The lower limit of integration.
    b : flt
        The upper limit of integration.
    mu : flt
        The std of the gaussian proposal density.
    N : int
        Number of samples.
    importance : int
        Importance sampling requested.
        
    Returns
    -------
    Integral : flt
        The estimated result.
    relErr : flt
        The relative error on the answer.
    acceptance : flt
        The acceptance ratio .
    '''
    Integral = 0 #variable to hold the integral estimate
    Integral2 = 0 #variable to hold running total for varience calculation
    rejection_ratio = 0 #variable to hold the rejection ratio
    samples = np.zeros(N) #variable so PDF can be checked on output
    x = np.random.uniform(a,b) #initialise coordinates
    for i in prange(N):
        x_trial = x + np.random.normal(0,mu) #choose trial step based on gaussian proposal density
        #metropolis algorithm
        if pdf(a,b,x_trial,importance)>=pdf(a,b,x,importance):
            #accpetance if point more likely
            x = x_trial #accept trial point
        else:
            dummy = np.random.rand()
            if dummy < pdf(a,b,x_trial,importance)/pdf(a,b,x,importance):
                #accpetance depending on probability of acceptance
                x = x_trial #accept trial point
            else:
                #point rejected
                rejection_ratio += 1./N #add to rejection ratio
        samples[i] = x #store samples
        Q = func(x)/pdf(a,b,x,importance) #the value of Q = f/P
        Integral += Q/N #add to integral in weighted sum
        Integral2 +=  Q**2/N #add to running toatal for calculateing varience
    relErr = math.sqrt((Integral2 - Integral**2)/float(N-1))/(Integral) #calculate relative error
    return (Integral,relErr,1-rejection_ratio,samples)

@njit
def montecarlo(func,a,b,mu,err,importance):
    '''
    Function to loop MCMC simulation to convergance.
    Returns the integral estimate.
    
    Parameters
    ----------
    func : function
        The function to be integrated.
    a : flt
        The lower limit of integration.
    b : flt
        The upper limit of integration.
    mu : flt
        The std of the gaussian proposal density.
    err : flt
        Desired relative error.
    importance : int
        Importance sampling requested.
        
    Raises
    ------
    TypeError : str
        If an invalid requested error is too low.
        
    Returns
    -------
    Integral : flt
        The estimated result.
    relErr : flt
        The relative error on the answer.
    acceptance : flt
        The acceptance ratio.
    N : int
        Number of samples.
    '''
    N = int(1e5) #set initial value of samples
    relerrs = [] #store errors
    Ns = [] #store sample size
    for i in prange(15):
        N*=5 #increase samples by factor of five each time
        Integral, relErr, acceptance, samples= metropolis(func,a,b,mu,N,importance) #run MCMC simulation
        Ns.append(N)
        relerrs.append(relErr)
        if relErr < err:
            #convergance criteria
            return (Integral, relErr, acceptance, N, samples, Ns, relerrs)
        print('Relative error after ',N,' samples: ',relErr) #prints error at time, reassures users for long running calculations.
    raise TypeError('Requested error too low to compute') #error raised if accuracy requested is too high

#%%
'''
Monte Carlo 1D Integration
'''
print('----------------------------------------------------------')
f = wavefunc
a = 0
b = 2
mu = 2
#no importance sampling
err = float(1e-3)#<--------------------Set Error 
importance = str('uniform')
start = time.time()
res = montecarlo(f,a,b,mu,err,importance)
end = time.time()
print('\n\nMONTE CARLO 1D INTEGRATION')
print('SHO GROUND STATE')
print('EXPECTED = 0.49766113\n')
print('\n',importance.upper(),' IMPORTANCE SAMPLING - ',1,'D SYSTEM',sep = '')
print('Time taken: ', end-start)
print('The computed integral is: ',res[0],'\nIts relative error is: ', res[1], '\nAcceptance rate: ', res[2],'\nSamples: ', res[3],'\n')
print('----------------------------------------------------------')
Nuniform = res[5]
relErruniform = res[6]

#%%
f = wavefunc
a = 0
b = 2
mu = 2
#linear importance sampling
err = float(1e-3)#<--------------------Set Error
importance = str('linear')
start = time.time()
res = montecarlo(f,a,b,mu,err,importance)
end = time.time()
print('\n\nMONTE CARLO 1D INTEGRATION')
print('SHO GROUND STATE')
print('EXPECTED = 0.49766113\n')
print('\n',importance.upper(),' IMPORTANCE SAMPLING - ',1,'D SYSTEM',sep = '')
print('Time taken: ', end-start)
print('The computed integral is: ',res[0],'\nIts relative error is: ', res[1], '\nAcceptance rate: ', res[2],'\nSamples: ', res[3],'\n\n ')
print('----------------------------------------------------------')
Nlinear = res[5]
relErrlinear = res[6]

#%%
'''
Relative error with/without importance sampling
'''

plt.figure()
plt.plot(Nuniform,relErruniform, label = 'Uniform Importance Sampling')
plt.plot(Nlinear,relErrlinear, label = 'Linear Importance Sampling')
plt.xlabel('Number of Samples')
plt.ylabel('Relative Accuracy')
plt.title('Relative Error Against Number of Samples')
plt.legend()