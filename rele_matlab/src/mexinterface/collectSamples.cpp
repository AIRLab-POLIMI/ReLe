#include "mex.h" /* Always include this */

#include <RandomGenerator.h>
#include <FileManager.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>

#include "CSDomainSettings.h"

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
    mexPrintf("  - params:       additional parameters for model or policy\n");
    mexPrintf(" OUTPUTS\n");
    mexPrintf("  - new_samples:  the set of episodess {state, action, nextstate, reward, absorb}\n");
    mexPrintf("  - dret:         expected discounted reward over per episode\n");
    mexPrintf("\n\n\n");
    mexPrintf("Available domains\n");
    mexPrintf("  - LQR\n");
    mexPrintf("    The policy is a standard Gaussian distribution with linear parametrization for the mean\n");
    mexPrintf("    * params\n");
    mexPrintf("      = policyParameters: mean parameters (vector)\n");
    mexPrintf("      = stddev: standard deviation of the policy (scalar)\n");
    mexPrintf("  - NLS\n");
    mexPrintf("    The policy is specifically designed for this domain\n");
    mexPrintf("    * params\n");
    mexPrintf("      = policyParameters: mean parameters (vector)\n");
    mexPrintf("  - Dam\n");
    mexPrintf("    The policy is a Gaussian distribution with logistic diagonal covariance with linear parametrization for the mean\n");
    mexPrintf("    * params\n");
    mexPrintf("      = policyParameters: policy parameters (vector)\n");
    mexPrintf("      = asVariance: asymptotic variance (vector)\n");
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

    CollectSamplesGateway(nlhs, plhs, nrhs, prhs);

    mxFree(domain_settings);
}
