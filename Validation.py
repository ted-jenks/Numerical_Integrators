import numpy as np
import time
import matplotlib.pyplot as plt
import P1_Newton_Coates as nc
import P2_Monte_Carlo as mc
import P3_Monte_Carlo_1D_Only as mc1d

'''
_________________________________________________________________________________
BEFORE RUNNING THIS MODULE MAKE SURE THE RESULT-GETTING CODE AT THE BOTTOM OF THE 
IMPORTED MODULES ARE COMMENTED OUT ELSE THEY WILL BE RUN.
_________________________________________________________________________________

This module holds some of the validation used to check my code. 
Chiefly this done by checking analytucally solveable integrals.
'''


#%%
'''
Newton-Coates 1D Integration Validation
Integral -2*x + 6 from 0 to 2
analytically = 8
'''

print('----------------------------------------------------------')
f = nc.validate
a = 0
b = 2
err = float(1e-6)#<--------------------Set Error
T = nc.trapezoidal(f,a,b,err)
print('\n\nNEWTON-COATES 1D INTEGRATION')
print('VALIDATION')
print('EXPECTED = 8')
print('\nTRAPEZOIDAL METHOD')
print('The computed integral is: ',T[0],'\nIts relative error is: ', T[1], '\nSamples: ', T[2])
S = nc.simpson(f,a,b,err)
print('\nSIMPSONS METHOD')
print('The computed integral is: ',S[0],'\nIts relative error is: ', S[1], '\nSamples: ', S[2],'\n\n ')
print('----------------------------------------------------------')
#%%
'''
Newton-Coates 1D Integration Validation
Integral sin(x) from 0 to 2pi
analytically = 0

THE INTEGRATOR WILL FAIL THIS INTEGRAL
This is because the answer is 0. This means that as the integrator gets close to the true answer,
the relative error tends to infinity and convergance is not met.
'''

print('----------------------------------------------------------')
f = nc.sin
a = 0
b = 2*np.pi
err = float(1e-6)#<--------------------Set Error
T = nc.trapezoidal(f,a,b,err)
print('\n\nNEWTON-COATES 1D INTEGRATION')
print('VALIDATION')
print('EXPECTED = 0')
print('\nTRAPEZOIDAL METHOD')
print('The computed integral is: ',T[0],'\nIts relative error is: ', T[1], '\nSamples: ', T[2])
S = nc.simpson(f,a,b,err)
print('\nSIMPSONS METHOD')
print('The computed integral is: ',S[0],'\nIts relative error is: ', S[1], '\nSamples: ', S[2],'\n\n ')
print('----------------------------------------------------------')

#%%
'''
Newton-Coates 3D Integration Validation
Integral 1/12*(x*y*z + x) with a = 0,0,0, b = 2,2,2
analytically = 8
'''

f = nc.validate3D
a = np.array([0,0,0])
b = np.array([4,2,1])
err = float(1e-4)#<--------------------Set Error
T = nc.trapezoidal_triple(f,a,b,err)
print('\n\nNEWTON-COATES 3D INTEGRATION')
print('VALIDATION')
print('EXPECTED = 2')
print('\nTRAPEZOIDAL METHOD')
print('The computed integral is: ',T[0],'\nIts relative error is: ', T[1], '\nSamples: ', T[2])
S = nc.simpson_triple(f,a,b,err)
print('\nSIMPSONS METHOD')
print('The computed integral is: ',S[0],'\nIts relative error is: ', S[1], '\nSamples: ', S[2],'\n\n ')
print('----------------------------------------------------------')


#%%
'''
Monte Carlo 1D Integration Validation
Integral -2*x + 6 from 0 to 2
analytically = 8
'''

print('\n\nMONTE CARLO 1D INTEGRATION')
print('VALIDATION')
print('EXPECTED = 8\n')
f = mc.validate
a = np.array([0])
b = np.array([2])
mu = 1
#no importance sampling
err = float(1e-3)#<--------------------Set Error
importance = str('uniform')
start = time.time()
res = mc.montecarlo(f,a,b,mu,err,importance)
end = time.time()
print('\n',importance.upper(),' IMPORTANCE SAMPLING - ',len(a),'D SYSTEM',sep = '')
print('Time taken: ', end-start)
print('The computed integral is: ',res[0],'\nIts relative error is: ', res[1], '\nAcceptance rate: ', res[2],'\nSamples: ', res[3],'\n')
print('----------------------------------------------------------')

#%%
'''
Monte Carlo 1D Integration Validation
Integral sin(x) from 0 to 2pi
analytically = 0
'''

print('\n\nMONTE CARLO 1D INTEGRATION')
print('VALIDATION')
print('EXPECTED = 0\n')
f = mc.sin
a = np.array([0])
b = np.array([2*np.pi])
mu = 1
#no importance sampling
err = float(1e-3)#<--------------------Set Error
importance = str('uniform')
start = time.time()
res = mc.montecarlo(f,a,b,mu,err,importance)
end = time.time()
print('\n',importance.upper(),' IMPORTANCE SAMPLING - ',len(a),'D SYSTEM',sep = '')
print('Time taken: ', end-start)
print('The computed integral is: ',res[0],'\nIts relative error is: ', res[1], '\nAcceptance rate: ', res[2],'\nSamples: ', res[3],'\n')
print('----------------------------------------------------------')

#%%
'''
Monte Carlo 3D Integration Validation
Integral 1/12*(x*y*z + x) with a = 0,0,0, b = 4,2,1
analytically = 8
'''

print('\n\nMONTE CARLO 3D INTEGRATION')
print('VALIDATION')
print('EXPECTED = 2\n')
f = mc.validate3D
a = np.array([0,0,0])
b = np.array([4,2,1])
mu = 1
#no importance sampling
err = float(1e-3)#<--------------------Set Error
importance = str('uniform')
start = time.time()
res = mc.montecarlo(f,a,b,mu,err,importance)
end = time.time()
print('\n',importance.upper(),' IMPORTANCE SAMPLING - ',len(a),'D SYSTEM',sep = '')
print('Time taken: ', end-start)
print('The computed integral is: ',res[0],'\nIts relative error is: ', res[1], '\nAcceptance rate: ', res[2],'\nSamples: ', res[3],'\n')

#%%
'''
Linear Sampling Plot
'''

plt.figure()
a = 0
b = 2
err = float(1e-4)
mu = 1.12
importance = str('linear')
res = mc1d.montecarlo(mc1d.wavefunc,a,b,mu,err,importance)
x = np.array([0,2])
y = np.zeros(2)
y[0]=mc1d.pdf(0,2,0,'linear')
y[1]=mc1d.pdf(0,2,2,'linear')
plt.title('Histogram of Linear Sampling Points')
plt.xlabel('x')
plt.ylabel('Frequancy')
plt.xlim(0,2)
plt.plot(x,y*820000,color = 'red',label = 'Input PDF')
plt.hist(res[4],bins = 30, label = 'Sample Distribution')
plt.legend()



