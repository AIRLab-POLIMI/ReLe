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

#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    const size_t statesNumber = 5;
    const size_t actionsNumber = 2;

    arma::cube R(actionsNumber, statesNumber, statesNumber, arma::fill::zeros);
    arma::cube Rsigma(actionsNumber, statesNumber, statesNumber, arma::fill::zeros);

    R(0, 1, 2) = 1;
    R(1, 3, 2) = 1;

    arma::cube P(actionsNumber, statesNumber, statesNumber);

    arma::mat P0(statesNumber, statesNumber);
    arma::mat P1(statesNumber, statesNumber);

    P0 << //
       0.2 << 0.8 << 0 << 0 << 0 << arma::endr //
       << 0 << 0.2 << 0.8 << 0 << 0 << arma::endr //
       << 0 << 0 << 0.2 << 0.8 << 0 << arma::endr //
       << 0 << 0 << 0 << 0.2 << 0.8 << arma::endr //
       << 0 << 0 << 0 << 0 << 1;

    P1 << //
       1 << 0 << 0 << 0 << 0 << arma::endr //
       << 0.8 << 0.2 << 0 << 0 << 0 << arma::endr //
       << 0 << 0.8 << 0.2 << 0 << 0 << arma::endr //
       << 0 << 0 << 0.8 << 0.2 << 0 << arma::endr //
       << 0 << 0 << 0 << 0.8 << 0.2;

    P.tube(arma::span(0), arma::span::all) = P0;
    P.tube(arma::span(1), arma::span::all) = P1;

    ReLe::FiniteMDP mdp(P, R, Rsigma, false, 0.9);
    //ReLe::e_Greedy policy;
    ReLe::Boltzmann policy;
    ReLe::SARSA agent(policy);
// 	ReLe::Q_Learning agent(policy);
    ReLe::Core<ReLe::FiniteAction, ReLe::FiniteState> core(mdp, agent);

    core.getSettings().episodeLenght = 10000;
    core.getSettings().logTransitions = false;
    cout << "starting episode" << endl;
    core.runEpisode();

}
