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

#include "rele/algorithms/td/SARSA.h"

using namespace std;
using namespace arma;

namespace ReLe
{

SARSA::SARSA(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha) :
    FiniteTD(policy, alpha)
{
}

void SARSA::initEpisode(const FiniteState& state, FiniteAction& action)
{
    sampleAction(state, action);
}

void SARSA::sampleAction(const FiniteState& state, FiniteAction& action)
{
    x = state.getStateN();
    u = policy(x);

    action.setActionN(u);
}

void SARSA::step(const Reward& reward, const FiniteState& nextState,
                 FiniteAction& action)
{
    size_t xn = nextState.getStateN();
    unsigned int un = policy(xn);
    double r = reward[0];

    double delta = r + task.gamma * Q(xn, un) - Q(x, u);
    Q(x, u) = Q(x, u) + alpha(x, u) * delta;

    //update action and state
    x = xn;
    u = un;

    //set next action
    action.setActionN(u);
}

void SARSA::endEpisode(const Reward& reward)
{
    //Last update
    double r = reward[0];
    double delta = r - Q(x, u);
    Q(x, u) = Q(x, u) + alpha(x, u) * delta;
}

SARSA::~SARSA()
{

}

SARSA_lambda::SARSA_lambda(ActionValuePolicy<FiniteState>& policy,
                           LearningRate& alpha,
                           bool accumulating) :
    FiniteTD(policy, alpha), accumulating(accumulating)
{
    lambda = 1;
}

void SARSA_lambda::initEpisode(const FiniteState& state, FiniteAction& action)
{
    sampleAction(state, action);
    Z.zeros();
}

void SARSA_lambda::sampleAction(const FiniteState& state, FiniteAction& action)
{
    x = state.getStateN();
    u = policy(x);

    action.setActionN(u);
}

void SARSA_lambda::step(const Reward& reward, const FiniteState& nextState,
                        FiniteAction& action)
{
    size_t xn = nextState.getStateN();
    unsigned int un = policy(xn);
    double r = reward[0];

    //compute TD error and eligibility trace
    double delta = r + task.gamma * Q(xn, un) - Q(x, u);
    if (accumulating)
        Z(x, u) = Z(x, u) + 1;
    else
        Z(x, u) = 1;

    //update action value function and eligibility trace
    Q = Q + alpha(x, u)* delta * Z;
    Z = task.gamma * lambda * Z;

    //update action and state
    x = xn;
    u = un;

    //set next action
    action.setActionN(u);
}

void SARSA_lambda::endEpisode(const Reward& reward)
{
    //Last update
    double r = reward[0];

    //compute TD error and eligibility trace
    double delta = r - Q(x, u);
    if (accumulating)
        Z(x, u) = Z(x, u) + 1;
    else
        Z(x, u) = 1;

    //update action value function
    Q = Q + alpha(x, u) * delta * Z;

}

SARSA_lambda::~SARSA_lambda()
{

}

void SARSA_lambda::init()
{
    FiniteTD::init();
    Z.zeros(task.statesNumber, task.actionsNumber);
}


}
