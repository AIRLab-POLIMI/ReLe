#include "rele/algorithms/td/LinearSARSA.h"

using namespace std;
using namespace arma;

namespace ReLe
{

LinearGradientSARSA::LinearGradientSARSA(Features& phi, ActionValuePolicy<DenseState>& policy,
        LearningRateDense& alpha)
    : LinearTD(phi, policy, alpha),
      prevQxu(arma::vec(task.stateDimensionality, arma::fill::zeros)),
      lambda(0.0),
      eligibility(Q.getParametersSize(), fill::zeros)
{
}

void LinearGradientSARSA::initEpisode(const DenseState& state, FiniteAction& action)
{
    this->eligibility = arma::vec(Q.getParametersSize(), arma::fill::zeros);
    sampleAction(state, action);

    unsigned int nStates = task.stateDimensionality;
    arma::vec regrInput(nStates + 1, arma::fill::zeros);
    for(unsigned int i = 0; i < nStates; i++)
        regrInput[i] = x[i];
    regrInput[nStates] = u;
    prevQxu = Q(regrInput);
}

void LinearGradientSARSA::sampleAction(const DenseState& state, FiniteAction& action)
{
    x = state;
    u = policy(x);

    action.setActionN(u);
}

void LinearGradientSARSA::step(const Reward& reward, const DenseState& nextState, FiniteAction& action)
{
    unsigned int nStates = task.stateDimensionality;
    unsigned int un = policy(nextState);

    // Prepare input for the regressor
    arma::vec regrInput(nStates + 1, arma::fill::zeros);

    // Q(xn, un)
    for(unsigned int i = 0; i < nStates; i++)
        regrInput[i] = nextState[i];
    regrInput[nStates] = un;
    arma::vec&& Qxnun = Q(regrInput);

    // Q(x,u)
    for(unsigned int i = 0; i < nStates; i++)
        regrInput[i] = x[i];
    regrInput[nStates] = u;
    arma::vec&& Qxu = Q(regrInput);

    double r = reward[0];

    double delta = r + task.gamma * Qxnun[0] - Qxu[0];

    Features_<arma::vec, true>& phi = Q.getFeatures();
    arma::vec temp = alpha(x, u) * (1 - task.gamma * this->lambda * this->eligibility.t() * phi(regrInput));
    this->eligibility = task.gamma * this->lambda * this->eligibility +
                        temp(0) * phi(regrInput);

    temp = alpha(x, u) * (prevQxu[0] - Qxu[0]);
    arma::vec deltaWeights = delta * this->eligibility + temp(0) * phi(regrInput);

    arma::vec regrWeights = Q.getParameters();
    regrWeights += deltaWeights;
    Q.setParameters(regrWeights);

    prevQxu = Qxnun;

    // update action and state
    x = nextState;
    u = un;

    // set next action
    action.setActionN(u);
}

void LinearGradientSARSA::endEpisode(const Reward& reward)
{
    // Last update
    unsigned int nStates = task.stateDimensionality;

    //Prepare input for the regressor
    arma::vec regrInput(nStates + 1);

    // Q(x,u)
    regrInput.subvec(0, nStates - 1) = x;
    regrInput[nStates] = u;
    arma::vec&& Qxu = Q(regrInput);

    double r = reward[0];
    double delta = r - Qxu[0];

    Features_<arma::vec, true>& phi = Q.getFeatures();
    arma::vec temp = alpha(x, u) * (1 - task.gamma * this->lambda * this->eligibility.t() * phi(regrInput));
    this->eligibility = task.gamma * this->lambda * this->eligibility +
                        temp(0) * phi(regrInput);

    temp = alpha(x, u) * (prevQxu[0] - Qxu[0]);
    arma::vec deltaWeights = delta * this->eligibility + temp(0) * phi(regrInput);

    // update regressor weights
    arma::vec regrWeights = Q.getParameters();
    regrWeights += deltaWeights;
    Q.setParameters(regrWeights);
}

void LinearGradientSARSA::setLambda(double lambda)
{
    this->lambda = lambda;
}

LinearGradientSARSA::~LinearGradientSARSA()
{

}

}//end namespace
