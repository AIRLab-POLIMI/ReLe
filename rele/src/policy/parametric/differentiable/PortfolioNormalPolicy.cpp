#include "parametric/differentiable/PortfolioNormalPolicy.h"
#include "RandomGenerator.h"

using namespace std;

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// PORTFOLIO NORMAL POLICY
///////////////////////////////////////////////////////////////////////////////////////

double PortfolioNormalPolicy::operator()(const arma::vec& state,
        typename action_type<FiniteAction>::const_type_ref action)
{
    arma::vec output = approximator(state);
    double prob = epsilon + (1 - 2 * epsilon) * exp(-0.01 * pow(output(0,0) - 10.0, 2));
    return (action == 1) ? prob : 1 - prob;
}

unsigned int PortfolioNormalPolicy::operator() (const arma::vec& state)
{
    double random = RandomGenerator::sampleUniform(0,1);
    arma::vec output = approximator(state);
    double prob = epsilon + (1 - 2 * epsilon) * exp(-0.01 * pow(output(0,0) - 10.0, 2));
    return (random <= prob) ? 1 : 0;

    // std::cout << "State: " << state[0] << " " << state[1] << " " << state[2] << " " << state[3] << " " << state[4] << " " << state[5] << std::endl;
    // arma::vec w = mpProjector->GetWeights();
    // std::cout << "Weights: " << w(0) << " " << w(1) << " " << w(2) << " " << w(3) << " " << w(4) << " " << w(5) << std::endl;
    // std::cout << "Evaluate(state): " << output << std::endl;
    // std::cout << "Prob(a=1): " << prob << std::endl;
}

arma::vec PortfolioNormalPolicy::diff(const arma::vec& state,
                                      typename action_type<FiniteAction>::const_type_ref action)
{
    return (*this)(state,action) * difflog(state,action);
}

arma::vec PortfolioNormalPolicy::difflog(const arma::vec& state,
        typename action_type<FiniteAction>::const_type_ref action)
{
    arma::vec output = approximator(state);
    double Exp = exp(0.01 * pow(output(0,0) - 10.0, 2));
    // the gradient is a vector of length nparams, where nparams is the parameter dimension
    unsigned nparams = approximator.getParametersSize();
    arma::vec gradient(nparams);
    for (unsigned i = 0; i < nparams; ++i)
    {

        gradient(i) = (1.0 - 2 * epsilon) / 50 * state[5-i] * (output(0,0) - 10.0);

        if(action == 1)
            gradient(i) /= -epsilon * Exp - 1 + 2 * epsilon;
        else
            gradient(i) /= (1.0 - epsilon) * Exp - 1 + 2 * epsilon;
    }
    return gradient;
}

arma::mat PortfolioNormalPolicy::diff2log(const arma::vec& state,
        typename action_type<FiniteAction>::const_type_ref action)
{
//TODO
    return arma::mat();
}

} //end namespace
