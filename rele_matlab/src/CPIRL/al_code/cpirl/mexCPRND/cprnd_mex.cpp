
#include "mex.h"

#include "cprnd_pure.h"
#include <ctime>

using namespace std;

#ifdef CPP11
#include <random>
std::mt19937 gen(time(0));
#endif

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  
  srand(time(0));

#define MATA prhs[1]
#define VECb prhs[2]
#define Np   prhs[0]
#define avgP plhs[0]
#define outStatus plhs[1]
#define Pmat plhs[2]

    /* Check for proper number of input and output arguments */
    if (nrhs < 3)
    {
        mexErrMsgIdAndTxt( "MATLAB:cprnd:minrhs",
                           "At least two input arguments required.");
    }
    if(nlhs > 3)
    {
        mexErrMsgIdAndTxt( "MATLAB:cprnd:maxlhs",
                           "Too many output arguments.");
    }

    /* Check inputs */


    double* A_mem = mxGetPr(MATA);
    double* b_mem = mxGetPr(VECb);
    int nbpoints  = mxGetScalar(Np);

    long int m = mxGetM(MATA);
    long int n = mxGetN(MATA);

    //     mat A_mat(A_mem, m, n);
    //     vec b_vec(b_mem, m);
    //
    //     cout << A_mat << endl;
    //     cout << b_vec << endl;


    //obtain points
    double* points = static_cast<double*>(malloc(sizeof(double)*nbpoints*n));
 
    try
    {
    cprnd_pure(A_mem, b_mem, m, n, nbpoints, points);
    }
    catch (int e)
    {
        if (nlhs > 0)
        {
            avgP = mxCreateDoubleMatrix(0, 0, mxREAL);
        }
        
        if (nlhs > 1)
        {    printf("sono qui");
             outStatus = mxCreateDoubleScalar(-1);
        }
        if (nlhs > 2)
        {
            Pmat = mxCreateDoubleMatrix(0, 0, mxREAL);
        }
        return;
    }
  
    if (nlhs > 0)
    {
        avgP = mxCreateDoubleMatrix(n, 1, mxREAL);
        double* avgP_mem = mxGetPr(avgP);
        
        long int i, j;
        //compute mean value
        for (i = 0; i < nbpoints; ++i) {
            for (j = 0; j < n; ++j) {
                avgP_mem[j] += points[IDX(i,j,nbpoints)];
            }
        }
        
        
        for (j = 0; j < n; ++j) {
            avgP_mem[j] /= nbpoints;
        }
        
        
        if (nlhs > 1)
        {
            outStatus = mxCreateDoubleScalar(0);
        }
        
        if (nlhs > 2)
        {
            Pmat = mxCreateDoubleMatrix(nbpoints, n, mxREAL);
            double* Pmat_mem = mxGetPr(Pmat);
            double* memptr = points;
            for (unsigned int i = 0; i < nbpoints * n; ++i)
            {
                Pmat_mem[i] = memptr[i];
            }
        }
    }
    free(points);

}
