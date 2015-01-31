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

#include "FiniteMDP.h"
#include "td/SARSA.h"
#include "td/Q-Learning.h"
#include "Core.h"

#include "q_policy/e_Greedy.h"
#include "q_policy/Boltzmann.h"

#include "SimpleChainGenerator.h"

#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    ReLe::SimpleChainGenerator generator;
    generator.generate(5, 2);

    ReLe::FiniteMDP mdp = generator.getMPD(0.9);
    //ReLe::e_Greedy policy;
    ReLe::Boltzmann policy;
    ReLe::SARSA agent(policy);
    //ReLe::Q_Learning agent(policy);
    ReLe::Core<ReLe::FiniteAction, ReLe::FiniteState> core(mdp, agent);

    core.getSettings().episodeLenght = 10000;
    core.getSettings().logTransitions = false;
    cout << "starting episode" << endl;
    core.runEpisode();

}
