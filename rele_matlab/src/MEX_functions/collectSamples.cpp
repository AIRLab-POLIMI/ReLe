#include "mex.h" /* Always include this */

#include <RandomGenerator.h>
#include <FileManager.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>

#include "collectSamplesUtils.h"

using namespace std;
using namespace arma;
using namespace ReLe;




void help()
{
    mexPrintf(" [new_samples, dret] = collectSamples(domain,...\n\t\t\t nbEpisodes, maxSteps, gamma, [params])\n");
    mexPrintf(" INPUTS:\n");
    mexPrintf("  - domain:       domain name (e.g., HumanWalk)\n");
    mexPrintf("  - nbEpisodes:   maximum number of episodes\n");
    mexPrintf("  - maxSteps:     maximum number of steps\n");
    mexPrintf("  - gamma:        discount factor\n");
    mexPrintf("  - params:       additional parameters for model or policy (optinal) ACTUALLY NOT IMPLEMENTED\n");
    mexPrintf(" OUTPUTS\n");
    mexPrintf("  - new_samples:  the set of episodess {state, action, nextstate, reward, absorb}\n");
    mexPrintf("  - dret:         expected discounted reward over per episode\n");
}


void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
                 int nrhs, const mxArray *prhs[]) /* Input variables */
{
#define IN_DOMAIN     prhs[0]
#define IN_NBEPISODES prhs[1]
#define IN_MAXSTEPS   prhs[2]
#define IN_GAMMA      prhs[3]
#define IN_PAR_STRUCT prhs[4]

    if (nrhs < 5)
    {
        help();
        mexErrMsgTxt("collectSamples: missing input parameters!\n");
    }

    if (!mxIsChar(IN_DOMAIN))
        mexErrMsgTxt("collectSamples: argument 1 must be a string!\n");
    if (!mxIsScalar(IN_NBEPISODES))
        mexErrMsgTxt("collectSamples: argument 2 must be a scalar!\n");
    if (!mxIsScalar(IN_MAXSTEPS))
        mexErrMsgTxt("collectSamples: argument 3 must be a scalar!\n");
    if (!mxIsScalar(IN_GAMMA))
        mexErrMsgTxt("collectSamples: argument 4 must be a scalar!\n");
    if (!mxIsStruct(IN_PAR_STRUCT))
        mexErrMsgTxt("collectSamples: argument 5 must be a struct!\n");      

    char* domain_settings = mxArrayToString(IN_DOMAIN);

    if ((strcmp(domain_settings, "lqr") == 0) || (strcmp(domain_settings, "nls") == 0)
            || (strcmp(domain_settings, "dam") == 0)
       )
    {
        CollectSamplesInContinuousMDP(nlhs, plhs, nrhs, prhs);
    }
    else if(strcmp(domain_settings, "deep") == 0)
    {
        CollectSamplesInDenseMDP(nlhs, plhs, nrhs, prhs);
    }
    else
    {
        mexErrMsgTxt("collectSamples: Unknown settings!\n");
    }

    mxFree(domain_settings);
}
