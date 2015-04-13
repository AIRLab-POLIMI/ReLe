#include "mex.h" /* Always include this */

#include <RandomGenerator.h>
#include <FileManager.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>

using namespace std;
using namespace arma;
using namespace ReLe;


#define MEX_DATA_FIELDS(F) \
    const char* F [] = {"s", "a", "r", "nexts", "terminal"}


void help()
{
    mexPrintf(" [new_samples, dret] = mexCollectSamples(domain,...\n\t\t\t nbEpisodes, maxSteps)\n");
    mexPrintf(" INPUTS:\n");
    mexPrintf("  - domain:       domain name (e.g., HumanWalk)\n");
    mexPrintf("  - nbEpisodes:   maximum number of episodes\n");
    mexPrintf("  - maxSteps:     maximum number of steps\n");
    mexPrintf(" OUTPUTS\n");
    mexPrintf("  - new_samples:  the set of episodess {state, action, nextstate, reward, absorb}\n");
    mexPrintf("  - dret:      expected discounted reward over per episode\n");
}


void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
                 int nrhs, const mxArray *prhs[]) /* Input variables */
{
#define IN_DOMAIN     prhs[0]
#define IN_NBEPISODES prhs[1]
#define IN_MAXSTEPS   prhs[2]

#define SAMPLES  plhs[0]
#define DRETURN  plhs[1]

    char* domain_settings = mxArrayToString(IN_DOMAIN);

    if ((strcmp(domain_settings, "lqr") == 0) || (strcmp(domain_settings, "nls") == 0))
    {
        CollectSamplesInContinuousMDP(nrhs, prhs, nlhs, plhs);
    }
    else if(strcmp(domain_settings, "deep") == 0)
    {
        CollectSamplesInDenseMDP(nrhs, prhs, nlhs, plhs);
    }
    else
    {
        mexErrMsgTxt("mexCollectSamples: Unknown settings!\n");
    }

    NLS mdp;
    int dim = mdp.getSettings().continuosStateDim;

    //--- define policy (low level)
    DenseBasisVector basis;
    basis.generatePolynomialBasisFunctions(1,dim);
    delete basis.at(0);
    basis.erase(basis.begin());
    cout << "--- Mean regressor ---" << endl;
    cout << basis << endl;
    LinearApproximator meanRegressor(dim, basis);

    DenseBasisVector stdBasis;
    stdBasis.generatePolynomialBasisFunctions(1,dim);
    delete stdBasis.at(0);
    stdBasis.erase(stdBasis.begin());
    cout << "--- Standard deviation regressor ---" << endl;
    cout << stdBasis << endl;
    LinearApproximator stdRegressor(dim, stdBasis);
    arma::vec stdWeights(stdRegressor.getParametersSize());
    stdWeights.fill(0.5);
    stdRegressor.setParameters(stdWeights);


    NormalStateDependantStddevPolicy policy(&meanRegressor, &stdRegressor);
    //---

    PolicyEvalAgent
    <DenseAction,DenseState> agent(policy);

    double gamma = 0.99;
    ReLe::Core<DenseAction, DenseState> oncore(mdp, agent);
    MatlabCollectorStrategy<DenseAction, DenseState> strat = MatlabCollectorStrategy<DenseAction, DenseState>(gamma);
    oncore.getSettings().loggerStrategy = &strat;

    int horiz = mdp.getSettings().horizon;
    oncore.getSettings().episodeLenght = horiz;

    int nbTrajectories = 1e3;
    for (int n = 0; n < nbTrajectories; ++n)
        oncore.runTestEpisode();

    std::vector<MatlabCollectorStrategy<DenseAction,DenseState>::MatlabEpisode>& data = strat.data;

    int ds = data[0].dx;
    int da = data[0].du;
    int dr = data[0].dr;

    MEX_DATA_FIELDS(fieldnames);
    // return samples
    SAMPLES = mxCreateStructMatrix(data.size(), 1, 5, fieldnames);
    DRETURN = mxCreateDoubleMatrix(dr, data.size(), mxREAL);
    double* Jptr = mxGetPr(DRETURN);

    for (int i = 0, ie = data.size(); i < ie; ++i)
    {
        int steps = data[i].steps;

        mxArray* state_vector      = mxCreateDoubleMatrix(ds, steps, mxREAL);
//        mxSetM(state_vector, ds);
//        mxSetN(state_vector, steps);
//        mxSetData(state_vector, data[i].states);
        memcpy(mxGetPr(state_vector), data[i].states.memptr(), sizeof(double)*ds*steps);


        mxArray* nextstate_vector  = mxCreateDoubleMatrix(ds, steps, mxREAL);
//        mxSetM(nextstate_vector, ds);
//        mxSetN(nextstate_vector, steps);
//        mxSetData(nextstate_vector, data[i].nextstates);
        memcpy(mxGetPr(nextstate_vector), data[i].nextstates.memptr(), sizeof(double)*ds*steps);

        mxArray* action_vector     = mxCreateDoubleMatrix(da, steps, mxREAL);
//        mxSetM(action_vector, da);
//        mxSetN(action_vector, steps);
//        mxSetData(action_vector, data[i].actions);
        memcpy(mxGetPr(action_vector), data[i].actions.memptr(), sizeof(double)*da*steps);

        mxArray* reward_vector     = mxCreateDoubleMatrix(dr, steps, mxREAL);
//        mxSetM(reward_vector, dr);
//        mxSetN(reward_vector, steps);
//        mxSetData(reward_vector, data[i].rewards);
        memcpy(mxGetPr(reward_vector), data[i].rewards.memptr(), sizeof(double)*dr*steps);


        mxArray* absorb_vector     = mxCreateNumericMatrix(1, steps, mxINT32_CLASS, mxREAL);
//        mxSetM(absorb_vector, 1);
//        mxSetN(absorb_vector, steps);
//        mxSetData(absorb_vector, data[i].absorb);
        memcpy(mxGetPr(absorb_vector), data[i].absorb.memptr(), sizeof(signed char)*steps);


        mxSetFieldByNumber(SAMPLES, i, 0, state_vector);
        mxSetFieldByNumber(SAMPLES, i, 1, action_vector);
        mxSetFieldByNumber(SAMPLES, i, 2, reward_vector);
        mxSetFieldByNumber(SAMPLES, i, 3, nextstate_vector);
        mxSetFieldByNumber(SAMPLES, i, 4, absorb_vector);

        for (int oo = 0; oo < dr; ++oo)
            Jptr[i*dr+oo] = data[i].Jvalue[oo];
    }

    mxFree(domain);
}
