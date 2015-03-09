/*
 * rele,
 *
 *
 * Copyright (C) 2015  Davide Tateo & Matteo Pirotta
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

#include "LQR.h"
#include "policy_search/REPS/EpisodicREPS.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/LinearPolicy.h"
#include "BasisFunctions.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    LQR mdp(1,1);

    arma::vec mean(2);
    mean[0] = 10;
    arma::mat cov(2,2, arma::fill::eye);

    ParametricNormal dist(mean,cov);

    DenseBasisVector basis;
    basis.generatePolynomialBasisFunctions(1,1);
    LinearApproximator regressor(mdp.getSettings().continuosStateDim, basis);

    arma::vec init_params(2);
    init_params[0] = -0.1;
    init_params[1] = -0.1;

    regressor.setParameters(init_params);
    DetLinearPolicy<DenseState> policy(&regressor);

    EpisodicREPS agent(dist, policy);

    ReLe::Core<DenseAction, DenseState> core(mdp, agent);

    int episodes  = 10;
    for (int i = 0; i < episodes; i++)
    {
        core.getSettings().episodeLenght = 50;
        core.getSettings().logTransitions = false;
        cout << "starting episode" << endl;
        core.runEpisode();
    }

    return 0;
}
