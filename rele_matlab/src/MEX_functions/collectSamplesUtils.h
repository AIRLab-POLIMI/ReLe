#ifndef COLLECTSAMPLESUTILS_H_
#define COLLECTSAMPLESUTILS_H_

#include "mex.h"

#define MEX_DATA_FIELDS(F) \
    const char* F [] = {"s", "a", "r", "nexts", "terminal"}

void
CollectSamplesInContinuousMDP(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
);

#endif