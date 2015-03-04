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
#include "policy_search/PGPE/PGPE.h"
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
//    arma::arma_rng::set_seed(std::time(0));
    LQR mdp(1,1);
    DenseState s(1);
    mdp.getInitialState(s);

    arma::vec mean(2);
    mean[0] = 10;
    arma::mat cov(2,2, arma::fill::eye);

    ParametricNormal dist(mean,cov);

//    std::map<int, int> hist;
//    for(int n=0; n<10000; ++n) {
//        vec theta = dist();
//        int id = std::round(theta[0]);
//        ++hist[id];
//    }
//    for(auto p : hist) {
//        std::cout << std::fixed << std::setprecision(1) << std::setw(2)
//                  << p.first << ' ' << std::string(p.second/200, '*') << '\n';
//    }

    DenseBasisVector* basis = new DenseBasisVector();
    basis->GeneratePolynomialBasisFunctions(1,1);
    cout << *basis << endl;
    LinearApproximator regressor(mdp.getSettings().continuosStateDim, basis);

    arma::vec init_params(2);
    init_params[0] = -0.1;
    init_params[1] = -0.1;

    regressor.setParameters(init_params);
    DetLinearPolicy<DenseState> policy(&regressor);


    cout << s << endl;
    cout << "Action: " << policy(s) << endl;


    int nbepperpol = 10, nbpolperupd = 100;
    PGPE<DenseAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd,0.01);
    agent.setNormalization(true);

    ReLe::Core<DenseAction, DenseState> core(mdp, agent);

    int nbUpdates = 40;
    int episodes  = nbUpdates*nbepperpol*nbpolperupd;
    episodes = 10;
    for (int i = 0; i < episodes; i++)
    {
        core.getSettings().episodeLenght = 50;
        core.getSettings().logTransitions = false;
        cout << "starting episode" << endl;
        core.runEpisode();
    }

    return 0;
}
