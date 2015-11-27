#include "CSDomainSettings.h"

#include <DifferentiableNormals.h>
#include <Core.h>
#include <PolicyEvalAgent.h>
#include <parametric/differentiable/NormalPolicy.h>
#include <parametric/differentiable/GibbsPolicy.h>
#include <BasisFunctions.h>
#include <basis/PolynomialFunction.h>
#include <basis/IdentityBasis.h>
#include <basis/GaussianRbf.h>
#include <basis/ConditionBasedFunction.h>
#include <LQR.h>
#include <NLS.h>
#include <Dam.h>
#include <DeepSeaTreasure.h>
#include <policy_search/gradient/onpolicy/FunctionGradient.h>
#include <policy_search/gradient/onpolicy/FunctionHessian.h>
#include <features/DenseFeatures.h>
#include <features/SparseFeatures.h>
#include "RewardTransformation.h"

using namespace std;
using namespace ReLe;
using namespace arma;

//===============================================================================
// GENERATION OF THE DATASET
//-------------------------------------------------------------------------------
#define SAMPLES_GATHERING(ActionC, StateC, acdim, stdim) \
	mdp.setHorizon(maxSteps);\
        PolicyEvalAgent<ActionC,StateC> agent(policy);\
        ReLe::Core<ActionC, StateC> oncore(mdp, agent);\
        CollectorStrategy<ActionC, StateC> collection;\
        oncore.getSettings().loggerStrategy = &collection;\
        oncore.getSettings().episodeLength = mdp.getSettings().horizon;\
        oncore.getSettings().testEpisodeN = nbEpisodes;\
        oncore.runTestEpisodes();\
        Dataset<ActionC,StateC>& data = collection.data;\
        int ds = stdim;\
        int da = acdim;\
        int dr = mdp.getSettings().rewardDim;\
        MEX_DATA_FIELDS(fieldnames);\
        SAMPLES = mxCreateStructMatrix(data.size(), 1, 5, fieldnames);\
        DRETURN = mxCreateDoubleMatrix(data.size(), dr, mxREAL);\
        double* Jptr = mxGetPr(DRETURN);\
        for (int i = 0, ie = data.size(); i < ie; ++i)\
        {\
            int nsteps = data[i].size();\
            mxArray* state_vector      = mxCreateDoubleMatrix(ds, nsteps, mxREAL);\
            mxArray* nextstate_vector  = mxCreateDoubleMatrix(ds, nsteps, mxREAL);\
            mxArray* action_vector     = mxCreateDoubleMatrix(da, nsteps, mxREAL);\
            mxArray* reward_vector     = mxCreateDoubleMatrix(dr, nsteps, mxREAL);\
            mxArray* absorb_vector     = mxCreateDoubleMatrix(1, nsteps, mxREAL);\
            double* states = mxGetPr(state_vector);\
            double* nextstates = mxGetPr(nextstate_vector);\
            double* actions = mxGetPr(action_vector);\
	    double* rewards = mxGetPr(reward_vector);\
            double* absorb = mxGetPr(absorb_vector);\
            int count = 0;\
            double df = 1.0;\
            arma::vec Jvalue(dr, arma::fill::zeros);\
            for (auto sample : data[i])\
            {\
                absorb[count] = 0;\
                for (int i = 0; i < ds; ++i)\
                {\
                    assigneStateWorker(states, count*ds+i, sample.x, i);\
                    assigneStateWorker(nextstates, count*ds+i, sample.xn, i);\
                }\
                for (int i = 0; i < da; ++i)\
                {\
                    assigneActionWorkerMEX(actions[count*da+i], sample.u, i);\
                }\
                for (int i = 0; i < dr; ++i)\
                {\
                    rewards[count*dr+i] = sample.r[i];\
                    Jvalue[i] += df*sample.r[i];\
                }\
                count++;\
                df *= gamma;\
            }\
            if (data[i][nsteps-1].xn.isAbsorbing())\
            {\
                absorb[nsteps-1] = 1;\
            }\
            mxSetFieldByNumber(SAMPLES, i, 0, state_vector);\
            mxSetFieldByNumber(SAMPLES, i, 1, action_vector);\
            mxSetFieldByNumber(SAMPLES, i, 2, reward_vector);\
            mxSetFieldByNumber(SAMPLES, i, 3, nextstate_vector);\
            mxSetFieldByNumber(SAMPLES, i, 4, absorb_vector);\
            for (int oo = 0; oo < dr; ++oo)\
                Jptr[i+oo*nbEpisodes] = Jvalue[oo] * mdp.getSettings().max_obj[oo];\
        }
        
#define GETSETTINGS \
    int nbEpisodes = mxGetScalar(IN_NBEPISODES);\
    int maxSteps   = mxGetScalar(IN_MAXSTEPS);\
    double gamma   = mxGetScalar(IN_GAMMA);

//===============================================================================
// INPUT and OUTPUT defines
//-------------------------------------------------------------------------------

// input
#define IN_DOMAIN     prhs[0]
#define IN_NBEPISODES prhs[1]
#define IN_MAXSTEPS   prhs[2]
#define IN_GAMMA      prhs[3]
#define IN_PAR_STRUCT prhs[4]

//outputs
#define SAMPLES  plhs[0]
#define DRETURN  plhs[1]
#define GRADS    plhs[2]
#define HESS     plhs[3]

//===============================================================================
// Utility function
//-------------------------------------------------------------------------------
inline void assigneStateWorker(double* val, int idx, FiniteState& state, int i)
{
    val[idx] = state.getStateN();
}

inline void assigneStateWorker(double* val, int idx, DenseState& state, int i)
{
    val[idx] = state[i];
}

inline void assigneActionWorkerMEX(double& val, FiniteAction& action, int i)
{
    val = action.getActionN() + 1;
}

inline void assigneActionWorkerMEX(double& val, DenseAction& action, int i)
{
    val = action[i];
}

//===============================================================================
// CollectSamplesGateway
//-------------------------------------------------------------------------------
void
CollectSamplesGateway(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
)
{
    // this allocate an array that must be destroied
    char* domain_settings = mxArrayToString(IN_DOMAIN);
    if (strcmp(domain_settings, "lqr") == 0)
    {
        lqr_domain_settings(nlhs, plhs, nrhs, prhs);
    }
    else if (strcmp(domain_settings, "nls") == 0)
    {
        nls_domain_settings(nlhs, plhs, nrhs, prhs);
    }
    else if (strcmp(domain_settings, "dam") == 0)
    {
        dam_domain_settings(nlhs, plhs, nrhs, prhs);
    }
    else if (strcmp(domain_settings, "deep") == 0)
    {
        deep_domain_settings(nlhs, plhs, nrhs, prhs);
    }
    else
    {
        mexErrMsgTxt("collectSamples: Unknown settings!\n");
    }

    // destroy char array
    mxFree(domain_settings);
}

//===============================================================================
// LQR Settings
//-------------------------------------------------------------------------------
void
lqr_domain_settings(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
)
{
    GETSETTINGS;
    // extract information from the struct
    mxArray* array = mxGetField(IN_PAR_STRUCT, 0, "dim");
    if (array == nullptr)
    {
        mexErrMsgTxt("LQR_DS: missing field dim!\n");
    }
    int dim = mxGetScalar(array);

    // get policy parameters
    array = mxGetField(IN_PAR_STRUCT, 0, "policyParameters");
    if (array == nullptr)
    {
        mexErrMsgTxt("LQR_DS: missing field policyParameters!\n");
    }
    int ncols = mxGetN(array);
    int nrows = mxGetM(array);
    if ( (ncols > 1 && nrows > 1) || (ncols == 0 && nrows == 0) )
    {
        mexErrMsgTxt("LQR_DS: wrong number of policy parameters!\n");
    }
    arma::vec policyParams(mxGetPr(array), ncols*nrows);

    // get standard deviation
    array = mxGetField(IN_PAR_STRUCT, 0, "stddev");
    if (array == nullptr)
    {
        mexErrMsgTxt("LQR_DS: missing field stddev!\n");
    }
    double stddev = mxGetScalar(array);

    //define LQR
    LQR mdp(dim,dim);

    //POLICY (Multivariate Gaussian with constant diagonal covariance)
    BasisFunctions basis;
    for (int ll = 0; ll < dim; ++ll)
        basis.push_back(new IdentityBasis(ll));
    SparseFeatures phi;
    phi.setDiagonal(basis);
    arma::mat cov (dim,dim, arma::fill::eye);
    cov *= stddev;
    MVNPolicy policy(phi,cov);
    if (policy.getParametersSize() != policyParams.n_elem)
        mexErrMsgTxt("LQR_DS: wrong number of policy parameters!\n");
    policy.setParameters(policyParams);

    //get dataset
    SAMPLES_GATHERING(DenseAction, DenseState, mdp.getSettings().continuosActionDim, mdp.getSettings().continuosStateDim)

    // additional outputs
    if (nlhs > 2)
    {
        int dp = policy.getParametersSize();
        GRADS = mxCreateDoubleMatrix(dp, dr, mxREAL);
        double* gptr = mxGetPr(GRADS);
        for (int i = 0; i < dr; ++i)
        {
            IndexRT rewardRegressor(i);
            GradientFromDataWorker<DenseAction,DenseState> gdw(data, policy, rewardRegressor, gamma);
            arma::vec g = gdw.GpomdpBaseGradient();
            for (int ll = 0; ll < dp; ++ll)
                gptr[i*dp+ll] = g[ll];
        }
    }
    if (nlhs > 3)
    {
        unsigned int dp = policy.getParametersSize();
        long unsigned int dims[] = {dr};
        HESS = mxCreateCellArray(1, dims);
        for (int i = 0; i < dr; ++i)
        {
            IndexRT rewardRegressor(i);
            HessianFromDataWorker<DenseAction,DenseState,MVNPolicy> gdw(data, policy, rewardRegressor, gamma);
            arma::mat h = gdw.ReinforceBaseHessian();

            mxArray* hmat = mxCreateDoubleMatrix(dp, dp, mxREAL);
            double* gptr = mxGetPr(hmat);
            memcpy(gptr, h.memptr(), sizeof(double)*dp*dp);
            mxSetCell(HESS, i, hmat);
        }
    }
}

//===============================================================================
// NLS Settings
//-------------------------------------------------------------------------------
void
nls_domain_settings(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
)
{
    GETSETTINGS;
    // extract information from the struct
    mxArray* array = mxGetField(IN_PAR_STRUCT, 0, "policyParameters");
    if (array == nullptr)
    {
        mexErrMsgTxt("NLS_DS: missing field policyParameters!\n");
    }
    int ncols = mxGetN(array);
    int nrows = mxGetM(array);
    if ( (ncols > 1 && nrows > 1) || (ncols == 0 && nrows == 0) )
    {
        mexErrMsgTxt("NLS_DS: wrong number of policy parameters!\n");
    }
    arma::vec policyParams(mxGetPr(array), ncols*nrows);

    //NLS domain
    NLS mdp;
    int dim = mdp.getSettings().continuosStateDim;

    //POLICY (NLS policy)
    BasisFunctions basis = IdentityBasis::generate(dim);
    DenseFeatures phi(basis);

    BasisFunctions stdBasis = IdentityBasis::generate(dim);
    DenseFeatures stdPhi(stdBasis);
    arma::vec stdWeights(stdPhi.rows());
    stdWeights.fill(0.5);

    NormalStateDependantStddevPolicy policy(phi, stdPhi, stdWeights);
    if (policy.getParametersSize() != policyParams.n_elem)
        mexErrMsgTxt("NLS_DS: wrong number of policy parameters!\n");
    policy.setParameters(policyParams);
    /*
            arma::vec pp(2);
            pp(0) = -0.4;
            pp(1) = 0.4;
            meanRegressor.setParameters(pp);*/

    //get dataset
    SAMPLES_GATHERING(DenseAction, DenseState, mdp.getSettings().continuosActionDim, mdp.getSettings().continuosStateDim)

    // additional outputs
    if (nlhs > 2)
    {
        int dp = policy.getParametersSize();
        GRADS = mxCreateDoubleMatrix(dp, dr, mxREAL);
        double* gptr = mxGetPr(GRADS);
        for (int i = 0; i < dr; ++i)
        {
            IndexRT rewardRegressor(i);
            GradientFromDataWorker<DenseAction,DenseState> gdw(data, policy, rewardRegressor, gamma);
            arma::vec g = gdw.GpomdpBaseGradient();
            for (int ll = 0; ll < dp; ++ll)
                gptr[i*dp+ll] = g[ll];
        }
    }
//     if (nlhs > 3)
//     {
//         unsigned int dp = policy.getParametersSize();
//         long unsigned int dims[] = {dr};
//         HESS = mxCreateCellArray(1, dims);
//         for (int i = 0; i < dr; ++i)
//         {
//             IndexRT rewardRegressor(i);
//             HessianFromDataWorker<DenseAction,DenseState,NormalStateDependantStddevPolicy> gdw(data, policy, rewardRegressor, gamma);
//             arma::mat h = gdw.GpomdpHessian();
// 
//             mxArray* hmat = mxCreateDoubleMatrix(dp, dp, mxREAL);
//             double* gptr = mxGetPr(hmat);
//             memcpy(gptr, h.memptr(), sizeof(double)*dp*dp);
//             mxSetCell(HESS, i, hmat);
//         }
//     }
}

//===============================================================================
// DAM Settings
//-------------------------------------------------------------------------------
void
dam_domain_settings(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
)
{
    GETSETTINGS;
    // extract information from the struct
    mxArray* array = mxGetField(IN_PAR_STRUCT, 0, "policyParameters");
    if (array == nullptr)
    {
        mexErrMsgTxt("DAM_DS: missing field policyParameters!\n");
    }
    int ncols = mxGetN(array);
    int nrows = mxGetM(array);
    if ( (ncols > 1 && nrows > 1) || (ncols == 0 && nrows == 0) )
    {
        mexErrMsgTxt("DAM_DS: wrong number of policy parameters!\n");
    }
    arma::vec policyParams(mxGetPr(array), ncols*nrows);

//         array = mxGetField(IN_PAR_STRUCT, 0, "asVariance");
//         if (array == nullptr)
//         {
//             mexErrMsgTxt("DAM_DS: missing field asVarince!\n");
//         }
//
//         ncols = mxGetN(array);
//         nrows = mxGetM(array);
//         if ( (ncols > 1 && nrows > 1) || (ncols == 0 && nrows == 0) )
//         {
//             mexErrMsgTxt("DAM_DS: wrong number of asymptotic variance elements!\n");
//         }
//         arma::vec as_variance(mxGetPr(array), ncols*nrows);

    //*** get mdp settings ***//
    //load default settings
    DamSettings settings;
    DamSettings::defaultSettings(settings);
    // number of rewards
    array = mxGetField(IN_PAR_STRUCT, 0, "nbRewards");
    if (array == nullptr)
    {
        mexErrMsgTxt("DAM_DS: missing field nbRewards!\n");
    }

    int nbRewards = mxGetScalar(array);
    if ( nbRewards < 1 || nbRewards > settings.rewardDim )
    {
        mexErrMsgTxt("DAM_DS: wrong number of rewards!\n");
    }
    settings.rewardDim = nbRewards;
    //penalize
    array = mxGetField(IN_PAR_STRUCT, 0, "penalize");
    if (array == nullptr)
    {
        mexErrMsgTxt("DAM_DS: missing field penalize!\n");
    }
    int penalize = mxGetScalar(array);
    settings.penalize = (penalize == 0) ? false : true;
    //initial state
    array = mxGetField(IN_PAR_STRUCT, 0, "initType");
    if (array == nullptr)
    {
        mexErrMsgTxt("DAM_DS: missing field initType!\n");
    }
    char* initStateType = mxArrayToString(array);
    if (strcmp(initStateType, "random") == 0)
    {
        settings.initial_state_type = DamSettings::initType::RANDOM;
    }
    else if (strcmp(initStateType, "random_discrete") == 0)
    {
        settings.initial_state_type = DamSettings::initType::RANDOM_DISCRETE;
    }
    else
    {
        mexErrMsgTxt("DAM_DS: available initType are 'random' and 'random_discrete'!\n");
    }
    Dam mdp(settings);

    PolynomialFunction *pf = new PolynomialFunction();
    //         GaussianRbf* gf1 = new GaussianRbf(-4, 1/0.0013, false);
//         GaussianRbf* gf2 = new GaussianRbf(52, 1/0.0013, false);
//         GaussianRbf* gf3 = new GaussianRbf(108, 1/0.0013, false);
//         GaussianRbf* gf4 = new GaussianRbf(164, 1/0.0013, false);
    GaussianRbf* gf1 = new GaussianRbf(-20, 60, true);
    GaussianRbf* gf2 = new GaussianRbf(50, 60, true);
    GaussianRbf* gf3 = new GaussianRbf(120, 60, true);
    GaussianRbf* gf4 = new GaussianRbf(190, 60, true);
//         GaussianRbf* gf1 = new GaussianRbf(0, 50, true);
//         GaussianRbf* gf2 = new GaussianRbf(50, 20, true);
//         GaussianRbf* gf3 = new GaussianRbf(120, 40, true);
//         GaussianRbf* gf4 = new GaussianRbf(160, 50, true);
    BasisFunctions basis;
    basis.push_back(pf);
    basis.push_back(gf1);
    basis.push_back(gf2);
    basis.push_back(gf3);
    basis.push_back(gf4);

    DenseFeatures phi(basis);
//     MVNLogisticPolicy policy(phi, 50);
    MVNDiagonalPolicy policy(phi);
    if (policy.getParametersSize() != policyParams.n_elem)
        mexErrMsgTxt("DAM_DS: wrong number of policy parameters!\n");
    policy.setParameters(policyParams);

    SAMPLES_GATHERING(DenseAction, DenseState, mdp.getSettings().continuosActionDim, mdp.getSettings().continuosStateDim)

    if (nlhs > 2)
    {
        int dp = policy.getParametersSize();
        GRADS = mxCreateDoubleMatrix(dp, dr, mxREAL);
        double* gptr = mxGetPr(GRADS);
        for (int i = 0; i < dr; ++i)
        {
            IndexRT rewardRegressor(i);
            GradientFromDataWorker<DenseAction,DenseState> gdw(data, policy, rewardRegressor, gamma);
            arma::vec g = gdw.GpomdpBaseGradient();
            for (int ll = 0; ll < dp; ++ll)
                gptr[i*dp+ll] = g[ll];
        }
    }
    if (nlhs > 3)
    {
        unsigned int dp = policy.getParametersSize();
        long unsigned int dims[] = {dr};
        HESS = mxCreateCellArray(1, dims);
        for (int i = 0; i < dr; ++i)
        {
            IndexRT rewardRegressor(i);
            HessianFromDataWorker<DenseAction,DenseState,MVNDiagonalPolicy> gdw(data, policy, rewardRegressor, gamma);
            arma::mat h = gdw.ReinforceBaseHessian();

            mxArray* hmat = mxCreateDoubleMatrix(dp, dp, mxREAL);
            double* gptr = mxGetPr(hmat);
            memcpy(gptr, h.memptr(), sizeof(double)*dp*dp);
            mxSetCell(HESS, i, hmat);
        }
    }
    mxFree(initStateType);
}

//===============================================================================
// DEEP Settings
//-------------------------------------------------------------------------------

///////////////////////////////////////////////////////////// USED FOR DEEP SEA TREASURE

class deep_2state_identity: public BasisFunction
{
    double operator()(const arma::vec& input)
    {
        return ((input[0] == 1) && (input[1] == 1))?1:0;
    }
    void writeOnStream(std::ostream& out)
    {
        out << "deep_2state" << endl;
    }
    void readFromStream(std::istream& in) {}
};

class deep_state_identity: public BasisFunction
{
    double operator()(const arma::vec& input)
    {
        return (input[0] == 1)?1:0;
    }
    void writeOnStream(std::ostream& out)
    {
        out << "deep_state" << endl;
    }
    void readFromStream(std::istream& in) {}
};
/////////////////////////////////////////////////////////////

void
deep_domain_settings(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
)
{
    GETSETTINGS;
    // extract information from the struct
    mxArray* array = mxGetField(IN_PAR_STRUCT, 0, "policyParameters");
    if (array == nullptr)
    {
        mexErrMsgTxt("DEEP_DS: missing field policyParameters!\n");
    }
    int ncols = mxGetN(array);
    int nrows = mxGetM(array);
    if ( (ncols > 1 && nrows > 1) || (ncols == 0 && nrows == 0) )
    {
        mexErrMsgTxt("DEEP_DS: wrong number of policy parameters!\n");
    }
    arma::vec policyParams(mxGetPr(array), ncols*nrows);

    DeepSeaTreasure mdp;
    vector<FiniteAction> actions;
    for (int i = 0; i < mdp.getSettings().finiteActionDim; ++i)
        actions.push_back(FiniteAction(i));

    //--- policy setup
    PolynomialFunction* pf0 = new PolynomialFunction();
    vector<unsigned int> dim = {0,1};
    vector<unsigned int> deg = {1,0};
    PolynomialFunction* pfs1 = new PolynomialFunction(dim,deg);
    deg = {0,1};
    PolynomialFunction* pfs2 = new PolynomialFunction(dim,deg);
    deg = {1,1};
    PolynomialFunction* pfs1s2 = new PolynomialFunction(dim, deg);
    deep_2state_identity* d2si = new deep_2state_identity();
    deep_state_identity* dsi = new deep_state_identity();

    BasisFunctions bfs;

    for (int i = 0; i < actions.size() -1; ++i)
    {
        bfs.push_back(new AndConditionBasisFunction(pf0,2,i));
        bfs.push_back(new AndConditionBasisFunction(pfs1,2,i));
        bfs.push_back(new AndConditionBasisFunction(pfs2,2,i));
        bfs.push_back(new AndConditionBasisFunction(pfs1s2,2,i));
        bfs.push_back(new AndConditionBasisFunction(d2si,2,i));
        bfs.push_back(new AndConditionBasisFunction(dsi,2,i));
    }

    DenseFeatures phi(bfs);

    ParametricGibbsPolicy<DenseState> policy(actions, phi, 1);
    //---
    if (policy.getParametersSize() != policyParams.n_elem)
        mexErrMsgTxt("DEEP_DS: wrong number of policy parameters!\n");
    policy.setParameters(policyParams);

    SAMPLES_GATHERING(FiniteAction, DenseState, 1, mdp.getSettings().continuosStateDim)


    if (nlhs > 2)
    {
        int dp = policy.getParametersSize();
        GRADS = mxCreateDoubleMatrix(dp, dr, mxREAL);
        double* gptr = mxGetPr(GRADS);
        for (int i = 0; i < dr; ++i)
        {
            IndexRT rewardRegressor(i);
            GradientFromDataWorker<FiniteAction,DenseState> gdw(data, policy, rewardRegressor, gamma);
            arma::vec g = gdw.GpomdpBaseGradient();
            for (int ll = 0; ll < dp; ++ll)
                gptr[i*dp+ll] = g[ll];
        }
    }
//         if (nlhs > 3)
//         {
//             unsigned int dp = policy.getParametersSize();
//             long unsigned int dims[] = {dr};
//             HESS = mxCreateCellArray(1, dims);
//             for (int i = 0; i < dr; ++i)
//             {
//                 IndexRT rewardRegressor(i);
//                 HessianFromDataWorker<FiniteAction,DenseState,MVNPolicy> gdw(data, policy, rewardRegressor, gamma);
//                 arma::mat h = gdw.GpomdpHessian();
//
//                 mxArray* hmat = mxCreateDoubleMatrix(dp, dp, mxREAL);
//                 double* gptr = mxGetPr(hmat);
//                 memcpy(gptr, h.memptr(), sizeof(double)*dp*dp);
//                 mxSetCell(HESS, i, hmat);
//             }
//         }
}