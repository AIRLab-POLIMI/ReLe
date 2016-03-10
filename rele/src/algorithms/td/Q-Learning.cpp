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

#include "rele/algorithms/td/Q-Learning.h"
#include "rele/utils/RandomGenerator.h"

using namespace std;
using namespace arma;

namespace ReLe
{

Q_Learning::Q_Learning(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha) :
    FiniteTD(policy, alpha)
{
}

void Q_Learning::initEpisode(const FiniteState& state, FiniteAction& action)
{
    sampleAction(state, action);
}

void Q_Learning::sampleAction(const FiniteState& state, FiniteAction& action)
{
    x = state.getStateN();
    u = policy(x);

    action.setActionN(u);
}

void Q_Learning::step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action)
{
    size_t xn = nextState.getStateN();
    double r = reward[0];
    double maxQxn;

    const rowvec& Qxn = Q.row(xn);
    maxQxn = Qxn.max();

    double delta = r + task.gamma * maxQxn - Q(x, u);
    Q(x, u) = Q(x, u) + alpha(x, u) * delta;

    //update action and state
    x = xn;
    u = policy(xn);

    //set next action
    action.setActionN(u);
}

void Q_Learning::endEpisode(const Reward& reward)
{
    //Last update
    double r = reward[0];
    double delta = r - Q(x, u);
    Q(x, u) = Q(x, u) + alpha(x, u) * delta;
}

Q_Learning::~Q_Learning()
{

}


}

