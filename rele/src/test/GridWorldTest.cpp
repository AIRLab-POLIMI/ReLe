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
#include "td/Q-Learning.h"
#include "Core.h"

#include <iostream>
#include "GridWorldGenerator.h"



using namespace std;

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        ReLe::GridWorldGenerator generator;
        generator.load(argv[1]);

        ReLe::FiniteMDP&& mdp = generator.getMPD(1.0);

        ReLe::e_Greedy policy;
        ReLe::SARSA_lambda agent(policy, false);
        //ReLe::SARSA agent(policy);
        //ReLe::Q_Learning agent(policy);

        ReLe::Core<ReLe::FiniteAction, ReLe::FiniteState> core(mdp, agent);

        core.getSettings().episodeLenght = 100000;
        core.getSettings().logTransitions = false;
        for (int i = 0; i < 2000; i++)
        {
            cout << "starting episode" << endl;
            core.runEpisode();
        }
    }
}
