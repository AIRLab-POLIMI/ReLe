#include "collecSamplesUtils.h"

#include <DifferentiableNormals.h>
#include <Core.h>
#include <PolicyEvalAgent.h>
#include <parametric/differentiable/NormalPolicy.h>
#include <BasisFunctions.h>
#include <basis/PolynomialFunction.h>

using namespace std;
using namespace ReLe;
using namespace arma;

#define IN_DOMAIN     prhs[0]
#define IN_NBEPISODES prhs[1]
#define IN_MAXSTEPS   prhs[2]
#define IN_GAMMA      prhs[3]

#define SAMPLES  plhs[0]
#define DRETURN  plhs[1]

void
CollectSamplesInContinuousMDP(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[]) /* Input variables */
)
{

    char* domain_settings = mxArrayToString(IN_DOMAIN);
    int nbEpisodes = mxGetScalar(IN_NBEPISODES);
    int maxSteps   = mxGetScalar(IN_MAXSTEPS);
    double gamma   = mxGetScalar(IN_GAMMA);

    if (strcmp(domain_settings, "lqr") == 0)
    {
        LQR mdp(1,1);
        PolynomialFunction* pf = new PolynomialFunction(1,1);
        cout << *pf << endl;
        DenseBasisVector basis;
        basis.push_back(pf);
        cout << basis << endl;
        LinearApproximator regressor(mdp.getSettings().continuosStateDim, basis);
        NormalPolicy policy(0.1, &regressor);


        //////////////////////////////////////////////// METTERE IN UNA DEFINE

        PolicyEvalAgent
        <DenseAction,DenseState> agent(policy);

        ReLe::Core<DenseAction, DenseState> oncore(mdp, agent);
        MatlabCollectorStrategy<DenseAction, DenseState> strat = MatlabCollectorStrategy<DenseAction, DenseState>(gamma);
        oncore.getSettings().loggerStrategy = &strat;

        int horiz = mdp.getSettings().horizon;
        oncore.getSettings().episodeLenght = horiz;

        int nbTrajectories = nbEpisodes;
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
            memcpy(mxGetPr(state_vector), data[i].states.memptr(), sizeof(double)*ds*steps);


            mxArray* nextstate_vector  = mxCreateDoubleMatrix(ds, steps, mxREAL);
            memcpy(mxGetPr(nextstate_vector), data[i].nextstates.memptr(), sizeof(double)*ds*steps);

            mxArray* action_vector     = mxCreateDoubleMatrix(da, steps, mxREAL);
            memcpy(mxGetPr(action_vector), data[i].actions.memptr(), sizeof(double)*da*steps);

            mxArray* reward_vector     = mxCreateDoubleMatrix(dr, steps, mxREAL);
            memcpy(mxGetPr(reward_vector), data[i].rewards.memptr(), sizeof(double)*dr*steps);


            mxArray* absorb_vector     = mxCreateNumericMatrix(1, steps, mxINT32_CLASS, mxREAL);
            memcpy(mxGetPr(absorb_vector), data[i].absorb.memptr(), sizeof(signed char)*steps);


            mxSetFieldByNumber(SAMPLES, i, 0, state_vector);
            mxSetFieldByNumber(SAMPLES, i, 1, action_vector);
            mxSetFieldByNumber(SAMPLES, i, 2, reward_vector);
            mxSetFieldByNumber(SAMPLES, i, 3, nextstate_vector);
            mxSetFieldByNumber(SAMPLES, i, 4, absorb_vector);

            for (int oo = 0; oo < dr; ++oo)
                Jptr[i*dr+oo] = data[i].Jvalue[oo];
        }
        ////////////////////////////////////////////////
    }
    else
    {
        mexErrMsgTxt("CollectSamplesInContinuousMDP: Unknown settings!\n");
    }



    mxFree(domain_settings);
}
