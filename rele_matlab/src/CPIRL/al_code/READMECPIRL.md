Code for Experiments


Contents:

1) Description of files in this (incremental) package
2) How-to: Run the CPIRL algorithm
4) How-to: Compare the projection algorithm, MWAL algorithm, LPAL algorithm and CPIRL algorithm on a driving simulator.

---

1) Description of files in this package

CPIRL.m: The algorithm from 

run_CPIRL.m: Wrapper for CPIRL algorithm.

---

2) How-to: Run the CPIRL algorithm

The comments in CPIRL.m explain its input and output parameters in detail. Basically, you give CPIRL.m a model of the enviroment, specified by THETA (transition function), F (feature function), and GAMMA (discount factor). You also give CPIRL.m an estimate of the expert's "feature expectations" in the vector E, and a number of iterations T to run. After CPIRL.m completes, the matrix PP contains T policies while the matrix contains WW weights. The last weight is the more accurate solutions found in T iterations.

*** NOTE: run_CPIRL.m require the CVX package and CPLEX for solving convex optimization problems. They can be found here:

http://www.stanford.edu/~boyd/cvx/
http://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/
