import numpy as np
import math
from numba import njit, prange
import matplotlib.pyplot as plt
import time

'''
This module contains the code for a n-dimensional Monte Carlo integrator with adaptive 
importance sampling.
The first time the code is run, Numba will compile it so this may take a little longer.
Therefore, I reccomend starting with low accuracies.

Result gathering code is at the bottom and can be changed for dfferent errors.
'''
#%%
@njit
def wavefunc(xarray):
    '''
    The ground state 1D SHO wavefunction.
    Returns the value of the function at x.
    
    Parameters
    ----------
    x : array (len = 1)
        The x value you want to evaluate the function for.
        
    Raises
    ------
    TypeError : str
        If xarray isn't one dimensional.
    
    Returns
    -------
    f : flt
        The value of the function at x.
    '''
    if len(xarray) != 1:
        raise TypeError('invalid limits for 1D function')
    x = xarray[0]
    return (1/math.sqrt(np.pi)) * math.exp(-x**2)


@njit
def wavefunc3D(xarray):
    '''
    The ground state 3D SHO wavefunction.
    Returns the value of the function at (x,y,z).
    
    Parameters
    ----------
    xarray : array (len = 3)
        The (x,y,z) values you want to evaluate the function for.
        
    Raises
    ------
    TypeError : str
        If xarray isn't three dimensional.
    
    Returns
    -------
    f : flt
        The values of the function at (x,y,z).
    '''
    if len(xarray) != 3:
        raise TypeError('invalid limits for 3D function')
    x = xarray[0]
    y = xarray[1]
    z = xarray[2]
    return (1/math.sqrt(np.pi))**3 * math.exp(-x**2-y**2-z**2)

@njit
def wavefunc3Dexcited(xarray):
    '''
    The excited state 3D SHO wavefunction.
    Returns the value of the function at (x,y,z).
    
    Parameters
    ----------
    xarray : array (len = 3)
        The (x,y,z) values you want to evaluate the function for.
        
    Raises
    ------
    TypeError : str
        If xarray isn't three dimensional.
    
    Returns
    -------
    f : flt
        The values of the function at (x,y,z).
    '''
    if len(xarray) != 3:
        raise TypeError('invalid limits for 3D function')
    x = xarray[0]
    y = xarray[1]
    z = xarray[2]
    return (1/math.sqrt(np.pi))**3 * math.exp(-x**2-y**2-z**2) * (x**2 + y**2)

@njit
def validate(xarray):
    '''
    An analytically solveble 1D integrand to test the integrator.
    Returns the value of the function at x.
    
    Parameters
    ----------
    x : array (len = 1)
        The x value you want to evaluate the function for.
    
    Returns
    -------
    f : flt
        The value of the function at x.
    '''
    x = xarray[0]
    return -2*x + 6

@njit()
def sin(xarray):
    '''
    An analytically solveble 1D integrand to test the integrator.
    Returns the value of the function at x.
    
    Parameters
    ----------
    x : array (len = 1)
        The x value you want to evaluate the function for.
    
    Returns
    -------
    f : flt
        The value of the function at x.
    '''
    x = xarray[0]
    return np.sin(x)

@njit
def validate3D(xarray):
    '''
    An analytically solveble 3D integrand to test the integrator.
    Returns the value of the function at (x,y,z).
    
    Parameters
    ----------
    xarray : array (len = 3)
        The (x,y,z) values you want to evaluate the function for.
    
    Returns
    -------
    f : flt
        The values of the function at (x,y,z).
    '''
    x = xarray[0]
    y = xarray[1]
    z = xarray[2]
    return 1/12*(x*y*z + x)

#%%

@njit
def pdf(a,b,x,importance, dimensions):
    '''
    A function of linear or uniform PDFs that depend on limits of integration.
    Function works for n dimensions.
    Returns the value of a linear or uniform PDF at x.
    
    Parameters
    ----------
    a : array
        The lower limits of integration.
    b : array
        The upper limits of integration.
    x : array
        The point being evaluated.
    importance : str
        The PDF desired.
    dimensions : int
        The number of dimensions of the PDF.
        
    Raises
    ------
    TypeError : str
        If an invalid PDF is requested.
        
    Returns
    -------
    f : flt
        The value of the PDF at x.
    '''
    for i in prange(dimensions):
        #if point is outside of integration limits return 0
        if x[i] < a[i] or x[i] > b[i]:
            return 0
    f = 1. #temp variable
    if importance == str('uniform'):
        #for uniform pdf return flat, normalised value
        for i in range(dimensions):
            f*=(b[i]-a[i])
        return 1/f
    if importance == str('linear'):
        #for linear PDF set up arrays for A and B
        A = np.zeros(dimensions)
        B = np.zeros(dimensions)
        for i in range(dimensions):
            #PDF normalised irrespective of integration range
            A[i] = -0.48 / ((b[i] - a[i]) * (-0.24 * (b[i] + a[i]) + 0.98))
            B[i] = 0.98 / ((b[i] - a[i]) * (-0.24 * (b[i] + a[i]) + 0.98))
            #multiply value in each dimension
            f *= (A[i] * x[i] + B[i])
        return f
    else:
        #if PDF isn't linear or uniform return an error
        raise TypeError('Invalid PDF')
        
@njit
def metropolis(func,a,b,mu,N,importance):
    '''
    Function to carry out MCMC simulation.
    Works for n dimensions.
    Returns the integral estimate.
    
    Parameters
    ----------
    func : function
        The function to be integrated.
    a : array
        The lower limits of integration.
    b : array
        The upper limits of integration.
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
    dimensions = len(a) #the number of dimensions of integration
    Integral = 0 #variable to hold the integral estimate
    Integral2 = 0 #variable to hold running total for varience calculation
    rejection_ratio = 0 #variable to hold the rejection ratio
    x = np.zeros(dimensions) #variable to hold coordinates
    for i in prange(dimensions):
        x[i] = np.random.uniform(a[i],b[i]) #initialise coordinates
    for i in prange(N):
        x_trial = np.zeros(dimensions) #variable to hold trial step
        for i in prange(dimensions):
            x_trial[i] = x[i] + np.random.normal(0,mu) #choose trial step based on gaussian proposal density
        #metropolis algorithm
        if pdf(a,b,x_trial,importance,dimensions)>=pdf(a,b,x,importance,dimensions):
            #accpetance if point more likely
            x = x_trial #accept trial point
        else:
            dummy = np.random.rand()
            if dummy < pdf(a,b,x_trial,importance,dimensions)/pdf(a,b,x,importance,dimensions):
                #accpetance depending on probability of acceptance
                x = x_trial #accept trial point
            else:
                #point rejected
                rejection_ratio += 1./N #add to rejection ratio
        Q = func(x)/pdf(a,b,x,importance,dimensions) #the value of Q = f/P
        Integral += Q/N #add to integral in weighted sum
        Integral2 +=  Q**2/N #add to running toatal for calculateing varience
    relErr = math.sqrt((Integral2 - Integral**2)/float(N-1))/(Integral) #calculate relative error
    return (Integral,relErr,1-rejection_ratio)

@njit
def montecarlo(func,a,b,mu,err,importance,adaptive=False):
    '''
    Function to loop MCMC simulation to convergance and perform adaptive sampling.
    Works for n dimensions.
    Returns the integral estimate.
    
    Parameters
    ----------
    func : function
        The function to be integrated.
    a : array
        The lower limits of integration.
    b : array
        The upper limits of integration.
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
    TypeError : str
        If len(a) =/= len(b).
    TypeError : str
        If a is larger than b.
        
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
    if len(a) != len(b):
        raise TypeError('dimensionality of limits must be the same')
    dimensions = len(a)
    for i in prange(dimensions):
        if a[i] >= b[i]:
            raise TypeError('lower limit must be lower than upper limit')
    N = int(1e5) #set initial value of samples
    for i in prange(15):
        N *=5 #increase samples by factor of five each time
        Integral, relErr, acceptance = metropolis(func,a,b,mu,N,importance) #run MCMC simulation
        #adaptive sampling changing mu to get optimal acceptance of ~ 0.23
        if importance == 'linear' and adaptive == True:
            #adaptive sampling for linear case
            #get acceptance ratio as close as possible to 0.234 by chaging mu
            if acceptance > 0.234:
                mu += 3*abs(acceptance-0.234) #if acceptance to high increase mu 
            else:
                mu -= 3*abs(acceptance-0.234) #if acceptance to low reduce mu 
        if relErr < err:
            #convergance criteria
            return (Integral, relErr, acceptance, N)
        print('Relative error after',N,'samples: ',relErr) #prints error at time, reassures users for long running calculations.
    raise TypeError('Requested error too low to compute') #error raised if accuracy requested is too high

#%%
'''
Monte Carlo 1D Integration
'''

print('----------------------------------------------------------')
f = wavefunc
a = np.array([0])
b = np.array([2])
mu = 1 #starting mu, will be adjusted by adaptive sampling
#no importance sampling
err = float(1e-3)#<--------------------Set Error 
importance = str('uniform')
start = time.time()
res = montecarlo(f,a,b,mu,err,importance)
end = time.time()
print('\n\nMONTE CARLO 1D INTEGRATION')
print('SHO GROUND STATE')
print('EXPECTED = 0.49766113\n')
print('\n',importance.upper(),' IMPORTANCE SAMPLING - ',len(a),'D SYSTEM',sep = '')
print('Time taken: ', end-start)
print('The computed integral is: ',res[0],'\nIts relative error is: ', res[1], '\nAcceptance rate: ', res[2],'\nSamples: ', res[3],'\n')
print('----------------------------------------------------------')
#%%

f = wavefunc
a = np.array([0])
b = np.array([2])
mu = 1 #starting mu, will be adjusted by adaptive sampling
#linear importance sampling
err = float(1e-3)#<--------------------Set Error
importance = str('linear')
start = time.time()
res = montecarlo(f,a,b,mu,err,importance,adaptive = True)
end = time.time()
print('\n\nMONTE CARLO 1D INTEGRATION')
print('SHO GROUND STATE')
print('EXPECTED = 0.49766113\n')
print('\n',importance.upper(),' IMPORTANCE SAMPLING - ',len(a),'D SYSTEM',sep = '')
print('Time taken: ', end-start)
print('The computed integral is: ',res[0],'\nIts relative error is: ', res[1], '\nAcceptance rate: ', res[2],'\nSamples: ', res[3],'\n\n ')
print('----------------------------------------------------------')

#%%
'''
Monte Carlo 3D Integration For Ground State
'''

f = wavefunc3D
a = np.array([0,0,0])
b = np.array([2,2,2])
mu = 1 #starting mu, will be adjusted by adaptive sampling
#no importance sampling
err = float(1e-3)#<--------------------Set Error
importance = str('uniform')
start = time.time()
res = montecarlo(f,a,b,mu,err,importance)
end = time.time()
print('\n\nMONTE CARLO 3D INTEGRATION')
print('SHO GROUND STATE')
print('EXPECTED = 0.123254\n')
print('\n',importance.upper(),' IMPORTANCE SAMPLING - ',len(a),'D SYSTEM',sep = '')
print('Time taken: ', end-start)
print('The computed integral is: ',res[0],'\nIts relative error is: ', res[1], '\nAcceptance rate: ', res[2],'\nSamples: ', res[3],'\n')
print('----------------------------------------------------------')
#%%
f = wavefunc3D
a = np.array([0,0,0])
b = np.array([2,2,2])
mu = 1 #starting mu, will be adjusted by adaptive sampling
#linear importance sampling
err = float(1e-3)#<--------------------Set Error
importance = str('linear')
start = time.time()
res = montecarlo(f,a,b,mu,err,importance)
end = time.time()
print('\n\nMONTE CARLO 3D INTEGRATION')
print('SHO GROUND STATE')
print('EXPECTED = 0.123254\n')
print('\n',importance.upper(),' IMPORTANCE SAMPLING - ',len(a),'D SYSTEM',sep = '')
print('Time taken: ', end-start)
print('The computed integral is: ',res[0],'\nIts relative error is: ', res[1], '\nAcceptance rate: ', res[2],'\nSamples: ', res[3],'\n\n ')
print('----------------------------------------------------------')

#%%
'''
Monte Carlo 3D Integration For Excited State
'''

f = wavefunc3Dexcited
a = np.array([0,0,0])
b = np.array([2,2,2])
mu = 1 #starting mu, will be adjusted by adaptive sampling
#no importance sampling
err = float(1e-3)#<--------------------Set Error
importance = str('uniform')
start = time.time()
res = montecarlo(f,a,b,mu,err,importance)
end = time.time()
print('\n\nMONTE CARLO 3D INTEGRATION')
print('SHO EXCITED STATE')
print('EXPECTED = 0.118136\n')
print('\n',importance.upper(),' IMPORTANCE SAMPLING - ',len(a),'D SYSTEM',sep = '')
print('Time taken: ', end-start)
print('The computed integral is: ',res[0],'\nIts relative error is: ', res[1], '\nAcceptance rate: ', res[2],'\nSamples: ', res[3],'\n')
print('----------------------------------------------------------')
#%%
f = wavefunc3Dexcited
a = np.array([0,0,0])
b = np.array([2,2,2])
mu = 1 #starting mu, will be adjusted by adaptive sampling
#linear importance sampling
err = float(1e-3)#<--------------------Set Error
importance = str('linear')
start = time.time()
res = montecarlo(f,a,b,mu,err,importance)
end = time.time()
print('\n\nMONTE CARLO 3D INTEGRATION')
print('SHO EXCITED STATE')
print('EXPECTED = 0.118136\n')
print('\n',importance.upper(),' IMPORTANCE SAMPLING - ',len(a),'D SYSTEM',sep = '')
print('Time taken: ', end-start)
print('The computed integral is: ',res[0],'\nIts relative error is: ', res[1], '\nAcceptance rate: ', res[2],'\nSamples: ', res[3],'\n\n ')
print('----------------------------------------------------------')

#%%
'''
Monte Carlo 1D Integration Histogram
'''

plt.figure()
f = wavefunc
a = np.array([0])
b = np.array([2])
mu = 1 #starting mu, will be adjusted by adaptive sampling
err = float(1e-2)
importance = str('linear')
integrals = []
N = 10 #<--------------- setting this higher will give a plot closer to a normal distribution
for i in range(N):
    res = montecarlo(f,a,b,mu,err,importance)
    integrals.append(res[0])
    N+=res[2]
plt.title('Histogram of 1D Monte Carlo Integrator Rsults')
plt.xlabel('Integral Estimate')
plt.ylabel('Frequancy')
plt.plot(0.49766113, N/10, 'x', label = 'True Value')
plt.legend()
plt.hist(integrals, bins = 25, ec='black')
