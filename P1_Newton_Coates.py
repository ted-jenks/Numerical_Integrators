import numpy as np
import math


'''
This module holds the ocde for one and three-dimensional Newton-Coates integration.

Result gathering code is at the bottom and can be changed for dfferent errors.
'''
#%%
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

def wavefunc3D(x,y,z):
    '''
    The ground state 3D SHO wavefunction.
    Returns the value of the function at (x,y,z).
    
    Parameters
    ----------
    x,y,z : flt
        The (x,y,z) values you want to evaluate the function for.
    
    Returns
    -------
    f : flt
        The values of the function at (x,y,z).
    '''
    return (1/math.sqrt(np.pi))**3 * math.exp(-x**2-y**2-z**2)

def wavefunc3Dexcited(x,y,z):
    '''
    The excited state 3D SHO wavefunction.
    Returns the value of the function at (x,y,z).
    
    Parameters
    ----------
    x,y,z : flt
        The (x,y,z) values you want to evaluate the function for.
    
    Returns
    -------
    f : flt
        The values of the function at (x,y,z).
    '''
    return (1/math.sqrt(np.pi))**3 * math.exp(-x**2-y**2-z**2) * (x**2 + y**2)

def validate(x):
    '''
    An analytically solveble 1D integrand to test the integrator.
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
    return -2*x + 6

def sin(x):
    '''
    An analytically solveble 1D integrand to test the integrator.
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
    return np.sin(x)

def validate3D(x,y,z):
    '''
    An analytically solveble 3D integrand to test the integrator.
    Returns the value of the function at (x,y,z).
    
    Parameters
    ----------
    x,y,z : flt
        The (x,y,z) values you want to evaluate the function for.
    
    Returns
    -------
    f : flt
        The values of the function at (x,y,z).
    '''
    return 1/12*(x*y*z + x)

#%%

def trapezoidal(f,a,b,err,forsimps = False):
    '''
    A function to run the extended trapezoidal rule.
    Returns the value of the integrand to a desired relative accuracy.
    
    Parameters
    ----------
    f : function
        The function you wish to integrate.
    a : flt
        The lower limit of integration.
    b : flt
        The upper limit of integration.
    err : flt
        The desired relative accuracy.
    forsimps : True/False
        Condition for if the integrator is being used for the simpson's rule.
        
    Raises
    ------
    TypeError : str
        If integral is equal to 0 and error diverges.
    
    Returns
    -------
    T : flt
        Integral estimate.
    test : flt
        The achieved relative accuracy.
    samples : int
        The number of samples taken to estimate the integral.
    Tprev : flt
        Previous integral estimate.
    '''
    if a >= b:
        raise TypeError('lower limit must be lower than upper limit')
    fa = f(a) #value of function at lower limit
    fb = f(b) #value of function at upper limit
    Tprev = 0 #setup varieable to hold previous estimation of integral
    h = b - a #caluculate initial step size
    T = h * 0.5 * (fa + fb) #first trapezoidal rule iteration
    j=1 #set iteration to 1
    samples = 2 #set samples taken to 2
    test = float('inf') #setup variable to hold the relative accuracy in a step
    while test > err: #condition for reaching convergance
        Tnext = 0.5 * T + 0.5 * h * sum(f((a+(2*i-1)/2)*h) for i in range(1,(2**(j-1)+1))) #calculation of integral by filling in midpoints and summing it to previous value 
        if forsimps == False: #if running as trapezoidal rule
            if abs(T) <= 1e-14 and j>10: #condition to avoid div by 0 in convergance calculation
                raise TypeError('integrator cannot converge for integrals equal to 0')    
            test = abs((Tnext - T) / T) #calculate relative accuracy
        if forsimps == True and j>1: #if running for simpson's rule
            if (4/3 * T - 1/3 * Tprev) <= 1e-14 and j>10: #condition to avoid div by 0 in convergance calculation
                raise TypeError('integrator cannot converge for integrals equal to 0')
            test = abs(((4/3 * Tnext - 1/3 * T) - (4/3 * T - 1/3 * Tprev))/(4/3 * T - 1/3 * Tprev)) #calculate relative accuracy for simpson's rule
        samples+=(2**(j-1)) #adds number of samples this iteration to count
        j += 1 #increases number of iterations by 1
        Tprev = T 
        T = Tnext
        h = h/2 #decrease stepsize by factor of 2
    return (T,test,samples,Tprev)

def simpson(f,a,b,err):
    '''
    A function to run the extended Simpson's rule using the trapezoidal rule.
    Returns the value of the integrand to a desired relative accuracy.
    
    Parameters
    ----------
    f : function
        The function you wish to integrate.
    a : flt
        The lower limit of integration.
    b : flt
        The upper limit of integration.
    err : flt
        The desired relative accuracy.
        
    Returns
    -------
    S : flt
        Integral estimate.
    test : flt
        The achieved relative accuracy.
    samples : int
        The number of samples taken to estimate the integral.
    '''
    T,test,samples,Tprev = trapezoidal(f, a, b, err, forsimps = True) #runs trapezoidal rule with forsimps = True
    S = 4/3*T - 1/3*Tprev #uses two final successive estimations to calculate simpson's rule
    return S,test,samples

def trapezoidal_triple(f, a, b, err, forsimps = False):
    '''
    A function to run the trapezoidal rule in 3D.
    Returns the value of the integrand to a desired relative accuracy.
    
    Parameters
    ----------
    f : function
        The function you wish to integrate.
    a : array (len = 3)
        The lower limits of integration.
    b : array (len = 3)
        The upper limits of integration.
    err : flt
        The desired relative accuracy.
        
    Raises
    ------
    TypeError : str
        If len(a) or len(b) don't equal 3.
        
    Returns
    -------
    T : flt
        Integral estimate.
    test : flt
        The achieved relative accuracy.
    samples : int
        The number of samples taken to estimate the integral.
    Tprev : flt
        Previous integral estimate.
    '''
    if len(a)!=3 or len(b)!=3:
        #raise error if limits have improper dimensionality
        raise TypeError('requires 3 upper and 3 lower limits of integration')
    def g(x,y):
        #runs trapezoidal rule over function
        return trapezoidal(lambda z: f(x, y, z), a[2], b[2], err, forsimps)[0]
    def p(x):
        #runs trapezoidal rule over g(x,y)
        return trapezoidal(lambda y: g(x,y), a[1], b[1], err,forsimps)[0]
    T,test,samples,Tprev = trapezoidal(p, a[0], b[0], err,forsimps) #run trapezoidal rule over p(x)
    samples = samples**3 #total samples will be samples in 1 dimension to the power of 3
    return T,test,samples,Tprev

def simpson_triple(f, a, b, err):
    '''
    A function to run the extended Simpson's rule in 3D using the trapezoidal rule.
    Returns the value of the integrand to a desired relative accuracy.
    
    Parameters
    ----------
    f : function
        The function you wish to integrate.
    a : array (len = 3)
        The lower limits of integration.
    b : array (len = 3)
        The upper limits of integration.
    err : flt
        The desired relative accuracy.
        
    Returns
    -------
    S : flt
        Integral estimate.
    test : flt
        The achieved relative accuracy.
    samples : int
        The number of samples taken to estimate the integral.
    '''
    T,test,samples,Tprev = trapezoidal_triple(f, a, b, err, forsimps = True) #runs trapezoidal rule with forsimps = True
    S = 4/3*T - 1/3*Tprev #uses two final successive estimations to calculate simpson's rule
    return S,test,samples


#%%
'''
Newton-Coates 1D Integration
'''

print('----------------------------------------------------------')
f = wavefunc
a = 0
b = 2
#trapezoidal rule
err = float(1e-6)#<--------------------Set Error
T = trapezoidal(f,a,b,err)
print('\n\nNEWTON-COATES 1D INTEGRATION')
print('SHO GROUND STATE')
print('EXPECTED = 0.49766113')
print('\nTRAPEZOIDAL METHOD')
print('The computed integral is: ',T[0],'\nIts relative error is: ', T[1], '\nSamples: ', T[2])
#simpson's rule
err = float(1e-8)#<--------------------Set Error
S = simpson(f,a,b,err)
print('\nSIMPSONS METHOD')
print('The computed integral is: ',S[0],'\nIts relative error is: ', S[1], '\nSamples: ', S[2],'\n\n ')
print('----------------------------------------------------------')

#%%
'''
Newton-Coates 3D Integration For Ground State
'''

f = wavefunc3D
a = np.array([0,0,0])
b = np.array([2,2,2])
#trapezoidal rule
err = float(1e-3)#<--------------------Set Error
T = trapezoidal_triple(f,a,b,err)
print('\n\nNEWTON-COATES 3D INTEGRATION')
print('SHO GROUND STATE')
print('EXPECTED = 0.123254')
print('\nTRAPEZOIDAL METHOD')
print('The computed integral is: ',T[0],'\nIts relative error is: ', T[1], '\nSamples: ', T[2])
#simpson's rule
err = float(1e-3)#<--------------------Set Error
S = simpson_triple(f,a,b,err)
print('\nSIMPSONS METHOD')
print('The computed integral is: ',S[0],'\nIts relative error is: ', S[1], '\nSamples: ', S[2],'\n\n ')
print('----------------------------------------------------------')

#%%
'''
Newton-Coates 3D Integration For Excited State
'''

f = wavefunc3Dexcited
a = np.array([0,0,0])
b = np.array([2,2,2])
#trapezoidal rule
err = float(1e-3)#<--------------------Set Error
T = trapezoidal_triple(f,a,b,err)
print('\n\nNEWTON-COATES 3D INTEGRATION')
print('SHO EXCITED STATE')
print('EXPECTED = 0.118136')
print('\nTRAPEZOIDAL METHOD')
print('The computed integral is: ',T[0],'\nIts relative error is: ', T[1], '\nSamples: ', T[2])
#simpson's rule
err = float(1e-3)#<--------------------Set Error
S = simpson_triple(f,a,b,err)
print('\nSIMPSONS METHOD')
print('The computed integral is: ',S[0],'\nIts relative error is: ', S[1], '\nSamples: ', S[2],'\n\n ')
print('----------------------------------------------------------')
