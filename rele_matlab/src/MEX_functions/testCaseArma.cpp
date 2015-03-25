#include <iostream>
#include <armadillo>
#include "mex.h"


using namespace std;
using namespace arma;

void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
                 int nrhs, const mxArray *prhs[]) /* Input variables */
{
    int n = 10;
    mat A(n,n);
    A.randn();
    A = A*A.t();
    A.save("prova.dat", raw_ascii);
    cout << A.t();
    cout << det(A);


// vec b = randn(n);
// 
// mat sol = solve(A,b);
}
