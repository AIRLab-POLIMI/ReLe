# ReLe
Reinforcement Learning Library of Politecnico di Milano

External dependencies
---------------------

To properly compile the ReLe library you must install
- [Armadillo](http://arma.sourceforge.net/)
- [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt)
- [Boost](http://www.boost.org/) (>= 1.53)

COMPILING
---------

The system can be build using the ros build tool `catkin`. Just create a catkin workspace, put the content of this repository in the src repository and run `catkin_make` to build the system.
check [this](http://ros.org/wiki/catkin/Tutorials/create_a_workspace) tutorial to get more info on catkin.

The ReLe core library can be build also using plain cmake (without catkin). To build the core library, use the following commands:

`mkdir ReLe` <br\>
`mkdir ReLe/build` <br\>
`mkdir ReLe/src` <br\>
`cd ReLe/src` <br\>
`git clone https://github.com/AIRLab-POLIMI/ReLe.git` <br\>
`cd ../build` <br\>
`cmake ../src/rele` <br\>
`make` <br\>

Ros features will be disabled.
Currently plain cmake installation is not supported.



