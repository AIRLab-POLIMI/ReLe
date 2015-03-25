#include "mex.h" /* Always include this */

#include <NLS.h>
#include <DifferentiableNormals.h>
#include <Core.h>
#include <PolicyEvalAgent.h>
#include <parametric/differentiable/NormalPolicy.h>
#include <policy_search/offpolicy/OffAlgorithm.h>
#include <BasisFunctions.h>
#include <basis/PolynomialFunction.h>
#include <basis/ConditionBasedFunction.h>
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


void help() {
    mexPrintf(" [new_samples, expdret, expuret, rethist] = mexCollectSamples(domain, mdp_config,...\n\t\t\t maxepisodes, maxsteps, policy, [isAvgReward])\n");
    mexPrintf(" INPUTS:\n");
    mexPrintf("  - domain:       domain name (e.g., HumanWalk)\n");
    mexPrintf("  - mdp_config:     the configuration of the mdp\n");
    mexPrintf("  - maxepisodes:  maximum number of episodes\n");
    mexPrintf("  - maxsteps:     maximum number of steps\n");
    mexPrintf("  - policy:       policy structure\n");
    mexPrintf("  - isAvgReward:  flag for average reward\n");
    mexPrintf(" OUTPUTS\n");
    mexPrintf("  - new_samples:  the set of episodess {state, action, nextstate, reward, absorb}\n");
    mexPrintf("  - expdret:      expected discounted reward over episodes\n");
    mexPrintf("  - expuret:      expected undiscounted reward over episodes\n");
    mexPrintf("  - rethist:      the return history, i.e., the expected return of each episode\n");
}


void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
                 int nrhs, const mxArray *prhs[]) /* Input variables */
{
#define SAMPLES  plhs[0]

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
    <DenseAction,DenseState,NormalStateDependantStddevPolicy > agent(policy);

    ReLe::Core<DenseAction, DenseState> oncore(mdp, agent);
    MatlabCollectorStrategy<DenseAction, DenseState> strat = MatlabCollectorStrategy<DenseAction, DenseState>();
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

	cout << "PROVAAAA" << endl;
	return;

    for (int i = 0, ie = data.size(); i < ie; ++i)
    {
	int steps = data[i].steps;

        mxArray* state_vector      = mxCreateDoubleMatrix(0, 0, mxREAL);
        mxSetM(state_vector, ds);
        mxSetN(state_vector, steps);
        mxSetData(state_vector, data[i].states);


        mxArray* nextstate_vector  = mxCreateDoubleMatrix(0, 0, mxREAL);
        mxSetM(nextstate_vector, ds);
        mxSetN(nextstate_vector, steps);
        mxSetData(nextstate_vector, data[i].nextstates);

        mxArray* action_vector     = mxCreateDoubleMatrix(0, 0, mxREAL);
        mxSetM(action_vector, ds);
        mxSetN(action_vector, steps);
        mxSetData(action_vector, data[i].actions);

        mxArray* reward_vector     = mxCreateDoubleMatrix(0, 0, mxREAL);
        mxSetM(reward_vector, dr);
        mxSetN(reward_vector, steps);
        mxSetData(reward_vector, data[i].rewards);


        mxArray* absorb_vector     = mxCreateNumericMatrix(0, 0, mxINT8_CLASS, mxREAL);
        mxSetM(absorb_vector, 1);
        mxSetN(absorb_vector, steps);
        mxSetData(absorb_vector, data[i].absorb);


        mxSetFieldByNumber(SAMPLES, i, 0, state_vector);
        mxSetFieldByNumber(SAMPLES, i, 1, action_vector);
        mxSetFieldByNumber(SAMPLES, i, 2, reward_vector);
        mxSetFieldByNumber(SAMPLES, i, 3, nextstate_vector);
        mxSetFieldByNumber(SAMPLES, i, 4, absorb_vector);

    }


}
