#ifndef CSDOMAINSETTINGS_H_
#define CSDOMAINSETTINGS_H_

#include "mex.h"

#define MEX_DATA_FIELDS(F) \
    const char* F [] = {"s", "a", "r", "nexts", "terminal"}

void
CollectSamplesGateway(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
);

void
lqr_domain_settings(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
);

void
nls_domain_settings(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
);

void
dam_domain_settings(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
);

void
deep_domain_settings(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
);



#endif