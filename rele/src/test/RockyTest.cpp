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

#include "Rocky.h"
#include "policy_search/REPS/EpisodicREPS.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/LinearPolicy.h"
#include "BasisFunctions.h"

#include <iostream>

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    Rocky rocky;

    int dim = rocky.getSettings().continuosStateDim;

    arma::vec mean(dim);
    arma::mat cov(dim, dim, arma::fill::eye);

    ParametricNormal dist(mean, cov);

    DenseBasisVector basis;
    basis.generatePolynomialBasisFunctions(1, dim - 1);
    LinearApproximator regressor(dim, basis);

    DetLinearPolicy<DenseState> policy(&regressor);

    EpisodicREPS agent(dist, policy);

    Core<DenseAction, DenseState> core(rocky, agent);

    int episodes = 40;
    for (int i = 0; i < episodes; i++)
    {
        core.getSettings().episodeLenght = 100000;
        cout << "starting episode" << endl;
        core.runEpisode();
    }

    return 0;

}
