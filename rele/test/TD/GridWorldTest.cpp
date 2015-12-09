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

#include "FiniteMDP.h"
#include "td/SARSA.h"
#include "td/DoubleQ-Learning.h"
#include "Core.h"

#include <iostream>
#include "GridWorldGenerator.h"



using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        GridWorldGenerator generator;
        generator.load(argv[1]);

        FiniteMDP&& mdp = generator.getMDP(1.0);

        e_Greedy policy;
        // SARSA_lambda agent(policy, false);
        // SARSA agent(policy);
        //Q_Learning agent(policy);
        DoubleQ_Learning agent(policy);

        auto&& core = buildCore(mdp, agent);

        core.getSettings().episodeLength = 100000;
        //core.getSettings().loggerStrategy = new WriteStrategy<FiniteAction, FiniteState>("/home/dave/prova.txt");
        //core.getSettings().loggerStrategy = new PrintStrategy<FiniteAction, FiniteState>(false);

        for(unsigned int i = 0; i < 2000; i++)
        {
            cout << endl << "### Starting episode " << i << " ###" << endl;
            core.runEpisode();
        }
    }
}
