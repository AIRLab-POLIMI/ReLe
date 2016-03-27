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

#include "rele/environments/GaussianRewardMDP.h"

#include "rele/utils/RandomGenerator.h"
#include "rele/utils/ArmadilloPDFs.h"

namespace ReLe
{

GaussianRewardMDP::GaussianRewardMDP(unsigned int dimension, double mu, double sigma,
                                     double gamma, unsigned int horizon) :
    ContinuousMDP(dimension, dimension, 1, true, true, gamma, horizon),
    A(dimension,dimension), B(dimension,dimension)
{
    initialize(dimension, mu, sigma);
}

GaussianRewardMDP::GaussianRewardMDP(arma::mat& A, arma::mat& B, arma::vec& mu, arma::mat& sigma,
                                     double gamma, unsigned int horizon) :
    ContinuousMDP(A.n_cols, B.n_cols, 1, true, true, gamma, horizon),
    A(A), B(B), mu(mu), sigma(sigma)
{

}

void GaussianRewardMDP::step(const DenseAction& action, DenseState& nextState,
                             Reward& reward)
{
    arma::vec& x = currentState;
    const arma::vec& u = action;

    x = A*x + B*u;

    nextState = currentState;
    nextState.setAbsorbing(false);

    reward[0] = mvnpdf(x, mu, sigma);

}

void GaussianRewardMDP::getInitialState(DenseState& state)
{
    for (unsigned int i = 0; i < mu.n_elem; ++i)
    {
        currentState(i) = RandomGenerator::sampleUniform(-10, 10);
    }

    currentState.setAbsorbing(false);

    state = currentState;
}

void GaussianRewardMDP::initialize(unsigned int dimensions, double mu_s, double sigma_s)
{
    A.eye();
    B.eye();

    mu.ones(dimensions);
    sigma.eye(dimensions, dimensions);

    sigma *= sigma_s;
    mu *= mu_s;

}




}
