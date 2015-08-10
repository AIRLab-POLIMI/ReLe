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

Remember to check the last supported version of GCC. For example, MATLAB R2015a supports up to GCC 4.7.x
To compile GCC 4.7 on *UBUNTU* you can use the following instructions
~~~~
#!/bin/sh
sudo apt-get install libmpfr-dev libgmp3-dev libmpc-dev flex bison libc6-dev-i386
sudo -K
rm -rf ~/Downloads/tmp
mkdir ~/Downloads/tmp
rm -rf ~/Downloads/gcc-4.7
mkdir ~/Downloads/gcc-4.7
cd ~/Downloads/tmp
wget http://ftp.gnu.org/gnu/gcc/gcc-4.7.4/gcc-4.7.4.tar.gz
tar xvzf gcc-4.7.4.tar.gz
cd gcc-4.7.4
mkdir build
cd build
../configure --disable-checking --enable-languages=c,c++ \
  --enable-multiarch --enable-shared --enable-threads=posix \
  --program-suffix=4.7 --with-gmp=/usr/local/lib --with-mpc=/usr/lib \
  --with-mpfr=/usr/lib --without-included-gettext --with-system-zlib \
  --with-tune=generic \
  --prefix=$HOME/Downloads/gcc-4.7
make -j4
make install
rm -rf ~/Downloads/tmp
~~~~

In *FEDORA* install the following packages and then use the same instructions above
~~~~
sudo yum install mpfr-devel gmp-devel libmpc-devel flex bison 
~~~~


First you need to compile the mex functions through MATLAB.
You need to move into *mexinterface/* where you can use the script *MEXMakefile* to compile the functions (you have to change the path to the compiler accordingly to your setup)
Then add such folder to the MATLAB search path.

### Provided functions
Here we report a brief descriptions of the functions. A more detailed explaination is provided in *mexinterface/README.md*.
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
