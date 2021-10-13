import numpy as np
import matplotlib.pyplot as plt

'''
This module generates the graphs showing error scaling with the number of samples.
Linear interpolation was used to estimate the number of samples required to reach
1e-6 relative accuracy for the 3D wavefunctions.
'''
#%%
'''
Error Scaling with Samples 1D Monte Carlo
'''

plt.figure()
N = np.array([125000000000,25000000000,5000000000,1562500000,62500000,331139,98113]) #sample number
err = np.array([1.1e-6,2.4596601427e-6,5.50085e-6,9.83871e-6,4.91957127e-5,0.0004697934,0.000864197]) #error achieved

ft = np.polyfit(err, N**-0.5, 1)
fit = np.poly1d(ft)
x = np.array([1e-6,0.0009])
plt.plot(err, N**-0.5, 'x', color = 'red')
plt.plot(x, fit(x), label = '1D Ground State', color = 'red')
plt.ylabel('N$^-$$^1$$^/$$^2$')
plt.xlabel('Error')
plt.title('Error Scaling with Samples Monte Carlo')

'''
Error Scaling with Samples 3D Monte Carlo Ground State
'''


N = np.array([1562500000,12500000,500000])
err = np.array([5.558e-6,6.204222265e-5,0.0003116077555])

ft = np.polyfit(err, N**-0.5, 1)
fit = np.poly1d(ft)
x = np.array([1e-6,0.0009])
plt.plot(err, N**-0.5, 'x', color = 'blue')
plt.plot(x, fit(x), color = 'blue',label = '3D Ground State')
plt.ylabel('N$^-$$^1$$^/$$^2$')
plt.xlabel('Error')
plt.legend()
print('Estimated N for 1e-6 relative error 3D Monte Carlo Ground State:', fit(1e-6)**-2)

#%%
'''
Error Scaling with Samples 3D Monte Carlo Excited State
'''

plt.figure()
N = np.array([1562500000,12500000,500000])
err = np.array([6.71381633777e-6,7.4983969969e-5,0.000375069071])

ft = np.polyfit(err, N**-0.5, 1)
fit = np.poly1d(ft)
x = np.array([1e-6,0.000375069071])
plt.plot(err, N**-0.5, 'x', label = 'Data')
plt.plot(x, fit(x), label = 'Fit')
plt.ylabel('N$^-$$^1$$^/$$^2$')
plt.xlabel('Error')
plt.title('Error Scaling with Samples 3D Monte Carlo Excited State')
print('Estimated N for 1e-6 relative error 3D Monte Carlo Excited State:', fit(1e-6)**-2)
plt.legend()

#%%
'''
Error Scaling with Samples 1D Trapezoidal Rule
'''

fig, axs = plt.subplots(2)
fig.tight_layout(pad=3.0) 
err = np.array([7.735169e-11,3.09413e-10,4.9505e-9,3.168e-7,5.0683667e-6,8.08544757e-5,0.00032035])
N = np.array([32769.,16385.,4097.,513.,129.,33.,17.])

ft = np.polyfit(err, N**-2, 1)
fit = np.poly1d(ft)
x = np.array([7.735169e-11,00.00032035])
axs[0].plot(x, fit(x),label = 'Fit', color = 'red')
axs[0].plot(err, N**-2, 'x', label = 'Data', color = 'blue')
axs[0].set_ylabel('N$^-$$^2$')
axs[0].set_xlabel('Error')
axs[0].set_title('Error Scaling with Samples 1D Trapezoidal Rule')
axs[0].legend()

'''
Error Scaling with Samples 3D Trapezoidal Rule
'''

err = np.array([5.068366739325193e-06,8.085447569e-5,0.00032035,00.00408349])
N = np.array([2146689.,35937.,453.,125.])

ft = np.polyfit(err, N**-2/3, 1)
fit = np.poly1d(ft)
x = np.array([1e-6,00.00408349])
axs[1].plot(x, fit(x),label = 'Fit', color = 'orange')
axs[1].plot(err, N**-2/3, 'x', label = 'Data',color = 'green')
axs[1].set_ylabel('N$^-$$^2$$^/$$^3$')
axs[1].set_xlabel('Error')
axs[1].set_title('Error Scaling with Samples 3D Trapezoidal Rule')
axs[1].legend()

#%%
'''
Error Scaling with Samples 1D Simpsons Rule
'''

plt.figure()
err = np.array([4.12565633e-9,6.601919451e-8,1.056671668e-6,1.687649668e-5])
N = np.array([129.,65.,33.,17.])

ft = np.polyfit(err, N**-4, 1)
fit = np.poly1d(ft)
x = np.array([7.735169e-11,1.687649668e-5])
plt.plot(x, fit(x),label = 'Fit', color = 'red')
plt.plot(err, N**-4, 'x', label = 'Data', color = 'blue')
plt.ylabel('N$^-$$^4$')
plt.xlabel('Error')
plt.title('Error Scaling with Samples 1D Simpsons Rule')
plt.legend()

#%%
'''
Error Scaling with Samples 3D Simpsons Rule
'''

plt.figure()
err = np.array([6.6019194e-8,1.056671668e-6,1.687649e-5])
N = np.array([274625.,35937.,4913.])
ft = np.polyfit(err, N**-4/3, 1)
fit = np.poly1d(ft)
x = np.array([7.735169e-11,1.687649668e-5])
plt.plot(err, N**-4/3, 'x', label = 'Data')
plt.plot(x, fit(x),label = 'Fit')
plt.ylabel('N$^-$$^4$$^/3$')
plt.xlabel('Error')
plt.title('Error Scaling with Samples 3D Simpsons Rule')
plt.legend()