# ReLe [![Documentation Status](https://readthedocs.org/projects/rele/badge/?version=latest)](http://rele.readthedocs.org/en/latest/?badge=latest) [![Build Status](http://131.175.56.232:8080/buildStatus/icon?job=ReLe-CI)](http://131.175.56.232:8080/buildStatus/icon?job=ReLe-CI)
**RE**inforcement **LE**arning Library of Politecnico di Milano

Tutorial and reference documentation is provided [here](http://rele.readthedocs.io/en/latest).

External dependencies
---------------------

To properly compile the ReLe library you must install
- [Armadillo](http://arma.sourceforge.net/) (>=6.1)
- [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt) (>= 2.4.2)
- [Boost](http://www.boost.org/) (>= 1.53)

### Linux
In many Linux distributions you can use the package manager to install them.

**Ubuntu**
```
apt-get install libboost-dev libboost-all-dev libnlopt-dev liblapack-dev libarmadillo-dev
```
**Fedora**
```
yum install boost-devel armadillo-devel openblas-devel lapack-devel
```

### Mac OS X
The easiest way to compile the library in a MAC OS X environment is to use [Homebrew](http://brew.sh/), which provides a package manager for OS X.
All the external libraries are available in the repository and can be installed using the following command
```
brew install boost homebrew/science/armadillo homebrew/science/nlopt
```

Homebrew provides also cmake package (`brew install cmake`).

### Windows
Refer to the dedicated [guide](README_WINVS.md).

COMPILING
---------

The system can be build using the ros build tool `catkin`. Just create a catkin workspace, put the content of this repository in the src repository and run `catkin_make` to build the system.
check [this](http://ros.org/wiki/catkin/Tutorials/create_a_workspace) tutorial to get more info on catkin.

The ReLe core library can be build also using plain cmake (without catkin). To build the core library, use the following commands:

```
mkdir ReLe
mkdir ReLe/build
mkdir ReLe/src
cd ReLe/src
git clone https://github.com/AIRLab-POLIMI/ReLe.git .
cd ../build
cmake ../src/rele
make
```

To install the library just use also the following command:

```
sudo make install
```

By default the library is installed in `/usr/local/`. You can change this by setting the cmake variable `${CMAKE_INSTALL_PREFIX}`

Ros features will be disabled.
