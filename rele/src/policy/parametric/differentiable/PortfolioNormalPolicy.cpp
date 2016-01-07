/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
 * Versione 1.0
 *
 * This file is part of rele.
 *
 * rele is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rele is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with rele.  If not, see <http://www.gnu.org/licenses/>.
 */


#include "rele/policy/parametric/differentiable/PortfolioNormalPolicy.h"
#include "rele/utils/RandomGenerator.h"

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
    double prob = epsilon + (1 - 2 * epsilon) * exp(-0.01 * (output(0) - 10.0) * (output(0) - 10.0));
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
    Features& basis = approximator.getFeatures();
    arma::mat features = basis(state);

    arma::vec output = approximator(state);
    double Exp = exp(0.01 * (output(0) - 10.0) * (output(0) - 10.0));
    // the gradient is a vector of length nparams, where nparams is the parameter dimension
    unsigned nparams = approximator.getParametersSize();
    arma::vec gradient(nparams);
    for (unsigned i = 0; i < nparams; ++i)
    {

        gradient(i) = (1.0 - 2 * epsilon) / 50 * features(i,0) * (output(0) - 10.0);

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

    int paramSize = this->getParametersSize();
    arma::mat hessian(paramSize,paramSize,arma::fill::zeros);
    int dm = approximator.getParametersSize();


    Features& basis = approximator.getFeatures();
    arma::mat features = basis(state);
    arma::vec output = approximator(state);
    double Exp = exp(0.01 * (output(0) - 10.0) * (output(0) - 10.0));

    for (unsigned i = 0; i < dm; ++i)
    {
        for (unsigned k = 0; k < dm; ++k)
        {

            if (action == 1)
            {
                double den = epsilon * (Exp - 2) + 1;
                double A = 0.04 * features(i,0) * features(k,0) * (epsilon - 0.5) / den;
                double B = 0.0008 * features(i,0) * features(k,0) * (epsilon - 0.5)
                           * epsilon * Exp * (output(0) - 10.0) * (output(0) - 10.0)
                           / (den * den);
                hessian(i,k) = A - B;
            }
            else
            {
                double den = (epsilon - 1) * Exp - 2 * epsilon + 1;
                double A = 0.04 * features(i,0) * features(k,0) * (epsilon - 0.5) / den;
                double B = 0.0008 * features(i,0) * features(k,0) * (epsilon - 1)
                           * (epsilon - 0.5) * Exp * (output(0) - 10.0) * (output(0) - 10.0)
                           / (den * den);
                hessian(i,k) = A - B;
            }
        }
    }

    return hessian;
}

} //end namespace
