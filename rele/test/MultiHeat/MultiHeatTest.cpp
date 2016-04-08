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

#include "rele/core/Core.h"
#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/FileManager.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include "rele/environments/MultiHeat.h"

using namespace std;
using namespace ReLe;
using namespace arma;


int main(int argc, char *argv[])
{
    RandomGenerator::seed(0);

    FileManager fm("multiheat", "test");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    MultiHeat mdp;

    DenseState cstate(mdp.getSettings().stateDimensionality);
    DenseState nextstate(mdp.getSettings().stateDimensionality);
    FiniteAction action;
    Reward reward(mdp.getSettings().rewardDimensionality);

    cstate(0) = 0;
    cstate(1) = 19.75;
    cstate(2) = 22;
    action.setActionN(0);

    mdp.getInitialState(cstate);
    mdp.setCurrentState(cstate);
    mdp.step(action, nextstate, reward);

    cout << "CurrentState: " << cstate << endl;
    cout << "Action: " << action << endl;
    cout << "NextState: " << nextstate << endl;
    cout << "Reward: " << reward[0] << endl;
    return 0;
}
