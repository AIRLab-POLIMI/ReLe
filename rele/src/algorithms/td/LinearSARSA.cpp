#include "td/LinearSARSA.h"

using namespace std;
using namespace arma;

namespace ReLe
{

LinearGradientSARSA::LinearGradientSARSA(LinearApproximator& la)
    : LinearTD(la), lambda(0.0), eligibility(la.getBasis().size(), fill::zeros)
{
}

void LinearGradientSARSA::initEpisode(const DenseState& state, FiniteAction& action)
{
    this->eligibility.zeros();
    sampleAction(state, action);
}

void LinearGradientSARSA::sampleAction(const DenseState& state, FiniteAction& action)
{
    x = state;
    u = policy(x);

    action.setActionN(u);
}

void LinearGradientSARSA::step(const Reward& reward, const DenseState& nextState, FiniteAction& action)
{
    unsigned int nstates = task.continuosStateDim;
    unsigned int un = policy(nextState);

    //Prepare input for the regressor
    vec regInput(nstates + 1);

    // Q(x,u)
    for (unsigned int i = 0; i < nstates; ++i)
    {
        regInput[i] = x[i];
    }
    regInput[nstates] = u;
    vec&& Qxu = Q(regInput);


    //Compute gradient dQ(x,u)
    BasisFunctions& basis = Q.getBasis();
    vec dQxu = basis(regInput);

    //Q(xn, un)
    for (unsigned int i = 0; i < nstates; ++i)
    {
        regInput[i] = nextState[i];
    }
    regInput[nstates] = un;
    vec&& Qxnun = Q(regInput);

    double r = reward[0];

    double delta = r + task.gamma * Qxnun[0] - Qxu[0];
    this->eligibility = task.gamma * this->lambda * this->eligibility + dQxu;

    vec deltaWeights = this->alpha * delta * this->eligibility;

    //TODO update
    vec& regWeights = Q.getParameters();
    regWeights += deltaWeights;

    //update action and state
    x = nextState;
    u = un;

    //set next action
    action.setActionN(u);
}

void LinearGradientSARSA::endEpisode(const Reward& reward)
{
    //Last update
    unsigned int nstates = task.continuosStateDim;

    //Prepare input for the regressor
    vec regInput(nstates + 1);

    // Q(x,u)
    for (unsigned int i = 0; i < nstates; ++i)
    {
        regInput[i] = x[i];
    }
    regInput[nstates] = u;
    vec&& Qxu = Q(regInput);


    //Compute gradient dQ(x,u)
    BasisFunctions& basis = Q.getBasis();
    vec dQxu = basis(regInput);


    double r = reward[0];
    double delta = r - Qxu[0];
    this->eligibility = task.gamma * this->lambda * this->eligibility + dQxu;

    vec deltaWeights = this->alpha * delta * this->eligibility;

    //update regressor weights
    vec& regWeights = Q.getParameters();
    regWeights += deltaWeights;

    //print statistics
    printStatistics();
}

void LinearGradientSARSA::endEpisode()
{
    //print statistics
    printStatistics();
}

LinearGradientSARSA::~LinearGradientSARSA()
{

}

void LinearGradientSARSA::printStatistics()
{
    cout << endl << endl << "### LINEAR SARSA ###";
    LinearTD::printStatistics();
}

}//end namespace
