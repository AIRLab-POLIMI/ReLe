Description
-----------

Contains all the functions to connect ReLe with Matlab. In particular there are:
- functions to read ReLe log files
- mex interfaces to call ReLe library from matlab
- matlab toolbox that contains almost all the algorithm that are implemented in ReLe

We leverage on the use of MiPS toolbox that we are helping in developing (https://github.com/sparisi/mips) but several extensions are provided here.


Code Structure
--------------

- *iodata* contains all the functions required to read ReLe log,
- *mexinterface* defines all the functions that allow Matlab to access ReLe functions,
- *mips* contains the Minimal Policy Search toolbox and several extensions to it
- *tests* contains auxiliary functions used to test ReLe implementation.


MEX interface
--------------------

This folder provides the access to ReLe functions via MATLAB.

First you need to compile the mex functions through MATLAB.
Remember to check the last supported version of GCC. For example, MATLAB R2015a supports up to GCC 4.7.x
You need to move into *mexinterface/* where you can use the script *MEXMakefile* to compile the functions.
Then add such folder to the MATLAB search path.

### Provided functions
#### collectSamples
~~~~
[new_samples, ret, G, H] = collectSamples(domain_settings, nbEpisodes, maxSteps, gamma, [params])
~~~~
To use *collectSamples* you need to define the domain settings. The domain settings defines all the elements required to simulate the environment.
In particular the following elements are defined in the settings:
1. domain that must be simulated with eventually all the associated parameters
2. policy
Finally, a last parameters can be given to the function. This input, named *params*, is usually a structure that contains additional parameters required by the given domain settings.

##### Inputs:
- name of the domain settings
- number of episodes
- maximum number of steps
- optional additional parameters

##### Outputs:
- trajectories
- expected return per trajectory
- gradient
- hessian

##### All the admissible calls are reported here:
~~~~
[new_samples, ret, G, H] = collectSamples(domain_settings, nbEpisodes, maxSteps, gamma)
[new_samples, ret, G, H] = collectSamples(domain_settings, nbEpisodes, maxSteps, gamma, params)
~~~~
