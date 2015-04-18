#include "td/LinearSARSA.h"

using namespace std;
using namespace arma;

namespace ReLe
{

LinearGradientSARSA::LinearGradientSARSA(ActionValuePolicy<DenseState>& policy, LinearApproximator& la)
    : LinearTD(policy, la), lambda(0.0), eligibility(la.getParameters().n_elem, fill::zeros),
      useReplacingTraces(false)
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
    Features& basis = Q.getBasis();
    mat dQxu = basis(regInput);


    //Q(xn, un)
    for (unsigned int i = 0; i < nstates; ++i)
    {
        regInput[i] = nextState[i];
    }
    regInput[nstates] = un;
    vec&& Qxnun = Q(regInput);

    double r = reward[0];

    double delta = r + task.gamma * Qxnun[0] - Qxu[0];
    //accumulatiog or replacing eligibility traces
    if (useReplacingTraces)
    {
        for (int i = 0; i < dQxu.n_elem; ++i)
        {
            if (dQxu[i] == 0)
                this->eligibility[i] =  task.gamma * this->lambda * this->eligibility[i];
            else
                this->eligibility[i] = dQxu[i];
        }
    }
    else
    {
        this->eligibility = task.gamma * this->lambda * this->eligibility + dQxu;
    }

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
    regInput.subvec(0, nstates - 1) = x;
    regInput[nstates] = u;
    vec&& Qxu = Q(regInput);


    //Compute gradient dQ(x,u)
    Features& basis = Q.getBasis();
    vec dQxu = arma::vectorise(basis(regInput));


    double r = reward[0];
    double delta = r - Qxu[0];
    this->eligibility = task.gamma * this->lambda * this->eligibility + dQxu;

    vec deltaWeights = this->alpha * delta * this->eligibility;

    //update regressor weights
    vec& regWeights = Q.getParameters();
    regWeights += deltaWeights;
}

LinearGradientSARSA::~LinearGradientSARSA()
{

}

}//end namespace
