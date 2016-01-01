# ReLe
Reinforcement Learning Library of Politecnico di Milano

[![Build Status](http://131.175.56.232:8080/job/ReLe-CI/badge/icon)](http://131.175.56.232:8080/job/ReLe-CI/)

External dependencies
---------------------

To properly compile the ReLe library you must install
- [Armadillo](http://arma.sourceforge.net/) (>=6.1)
- [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt)
- [Boost](http://www.boost.org/) (>= 1.53)

For example, in many Linux distributions you can use the package manager to install them.

**Ubuntu**
```
apt-get install libboost-dev libboost-all-dev libnlopt-dev liblapack-dev libarmadillo-dev
```
**Fedora**
```
yum install boost-devel armadillo-devel openblas-devel lapack-devel
```

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

Ros features will be disabled.
Currently plain cmake installation is not supported.



