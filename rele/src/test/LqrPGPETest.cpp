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

#include <iostream>

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
//    arma::arma_rng::set_seed(std::time(0));
    LQR mdp(1,1);
    DenseState s(1);
    mdp.getInitialState(s);
    cout << s << endl;

    ParametricNormal dist(1,1);
    vec theta = dist();
    cout << "Theta: " << theta;
//    LinearApproximator regressor(mdp.getSettings().continuosStateDim,mdp.getSettings().continuosActionDim);
//    DetLinearPolicy<DenseState> policy(&regressor);

//    int nbepperpol = 10, nbpolperupd = 100;
//    PGPE<DenseAction, DenseState> agent(&dist, &policy, nbepperpol, nbpolperupd,0.01);

//    ReLe::Core<DenseAction, DenseState> core(mdp, agent);

//    int nbUpdates = 40;
//    int episodes  = nbUpdates*nbepperpol*nbpolperupd;
//    for (int i = 0; i < episodes; i++)
//    {
//        core.getSettings().episodeLenght = 50;
//        core.getSettings().logTransitions = false;
//        cout << "starting episode" << endl;
//        core.runEpisode();
//    }

    return 0;
}
