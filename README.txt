Project 2 - Solving Quantum Systems Numerically												|
Edward Jenks																|
----------------------------------------------------------------------------------------------------------------------------------------|
																	|
The code submitted for my project was developed in python 3.8.										|
I used Spyder 4.1.5.															|
Whatever enviroment the code is run in, it is vital that the newest version of Numba is installed else the code will not run properly.  | 
																	|
________________________________________________________________________________________________________________________________________|

My Submission Contains 5 modules
1. Newton_Coates
2. Monte_Carlo
3. Monte_Carlo_1D_Only
4. Validation
5. Error Scaling

----------------
1. Newton_Coates
----------------
This module contains the code for the newton coates methods.
At the bottom, some code has been set up to retrieve the results so you can simply run the module.
The error desired is marked in this section and can be changed to any value.

There are four functions a user should run:
-trapezoidal(f,a,b,err)
	-runs the 1D trapezoidal rule
	-takes (function, lower limit, upper limit, desired error)
	-in this case a and b are floats
-simpson(f,a,b,err)
	-runs the 1D Simpson's rule
	-takes (function, lower limit, upper limit, desired error)
-trapezoidal_triple(f,a,b,err)
	-runs the 3D trapezoidal rule
	-takes (function, lower limit, upper limit, desired error)
	-in this case a and b are arrays
-simpson_triple(f,a,b,err)
	-runs the 3D Simpson's rule
	-takes (function, lower limit, upper limit, desired error)

--------------
2. Monte_Carlo
--------------
This module contains the code for the Monte Carlo methods.
The integrator can take a function of any dimensions.
At the bottom, some code has been set up to retrieve the results so you can simply run the module.
The error desired is marked in this section and can be changed to any value.
This module uses Numba.
The first time you run it it will take slightly longer - set accuracy low to start with.

There is one function the user should run:
-montecarlo(func,a,b,mu,err,importance,adaptive=False)
	-runs the Monte carlo integration.
	-takes (function, lower limit, upper limit, std deviation of proposal density, desired error, importance sampling = 'uniform' or 'linear', adaptive = True or False)
	-in this case, a and b are arrays with a length matching the dimensionality of the function.

As well as the results for each wavefunction, a histogram of results from multiple runs will be displayed.
Here, N (number of results) can be changed.
The larger N, the closer the histogram gets to a normal distribution.

----------------------
3. Monte_Carlo_1D_Only
----------------------
This module contains the code for the Monte Carlo 1D methods.
This is faster than the genral method and reccommended for weaker machines.
At the bottom, some code has been set up to retrieve the results so you can simply run the module.
The error desired is marked in this section and can be changed to any value.
This module uses Numba.
The first time you run it it will take slightly longer - set accuracy low to start with.

There is one function the user should run:
-montecarlo(func,a,b,mu,err,importance)
	-runs the Monte carlo integration.
	-takes (function, lower limit, upper limit, std deviation of proposal density, desired error, importance sampling = 'uniform' or 'linear', adaptive = True or False)
	-in this case, a and b are floats.

-------------
4. Validation
-------------
BEFORE RUNNING THIS MODULE MAKE SURE THE RESULT-GETTING CODE AT THE BOTTOM OF THE IMPORTED MODULES ARE COMMENTED OUT ELSE THEY WILL BE RUN.

This module holds some of the validation used to check my code. 
Chiefly this done by checking analytucally solveable integrals.
It also includes an analysis of importance sampling.
Simply running the module will return the results.

----------------
5. Error Scaling
----------------
This module generates the error scaling graphs shown and discussed in my report.
Simply running the module will generate the graphs.






