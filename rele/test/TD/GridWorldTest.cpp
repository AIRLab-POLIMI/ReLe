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


#include "rele/core/FiniteMDP.h"
#include "rele/algorithms/td/SARSA.h"
#include "rele/algorithms/td/DoubleQ-Learning.h"
#include "rele/algorithms/td/WQ-Learning.h"
#include "rele/core/Core.h"
#include "rele/generators/GridWorldGenerator.h"
#include "rele/policy/q_policy/e_Greedy.h"

#include <iostream>


using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    if(argc > 1)
    {
        GridWorldGenerator generator;
        generator.load(argv[1]);

        FiniteMDP&& mdp = generator.getMDP(0.9);

        e_Greedy policy;
        ConstantLearningRate alpha(0.2);
        //Q_Learning agent(policy, alpha);
        //DoubleQ_Learning agent(policy, alpha);
        WQ_Learning agent(policy, alpha);

        auto&& core = buildCore(mdp, agent);

        core.getSettings().episodeLength = 100000;
        core.getSettings().loggerStrategy = new PrintStrategy<FiniteAction, FiniteState>(false);

        for(unsigned int i = 0; i < 10000; i++)
        {
            cout << endl << "### Starting episode " << i << " ###" << endl;
            core.runEpisode();
        }
    }
}
