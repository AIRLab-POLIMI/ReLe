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

#include "rele/core/FiniteMDP.h"
#include "rele/algorithms/td/SARSA.h"
#include "rele/algorithms/td/Q-Learning.h"
#include "rele/core/Core.h"

#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/policy/q_policy/Boltzmann.h"

#include "rele/generators/SimpleChainGenerator.h"

#include "rele/approximators/basis/IdentityBasis.h"

#include <iostream>

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    SimpleChainGenerator generator;
    generator.generate(5, 2);

    FiniteMDP mdp = generator.getMDP(0.9);
    e_Greedy policy;
    ConstantLearningRate alpha(0.2);
    //Boltzmann policy;

    //SARSA agent(policy);
    Q_Learning agent(policy, alpha);

    Core<FiniteAction, FiniteState> core(mdp, agent);

    core.getSettings().episodeLength = 10000;
    core.getSettings().loggerStrategy = new PrintStrategy<FiniteAction, FiniteState>(false, true);
    cout << "starting episode" << endl;
    core.runEpisode();
    delete core.getSettings().loggerStrategy;

}
