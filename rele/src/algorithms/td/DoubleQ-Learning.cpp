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

/*
 * Written by: Alessandro Nuara, Carlo D'Eramo
 */

#include "rele/algorithms/td/DoubleQ-Learning.h"
#include "rele/utils/RandomGenerator.h"

using namespace std;
using namespace arma;


namespace ReLe
{

DoubleQ_Learning::DoubleQ_Learning(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha) :
    Q_Learning(policy, alpha)
{
}

void DoubleQ_Learning::initEpisode(const FiniteState& state, FiniteAction& action)
{
    sampleAction(state, action);
}

void DoubleQ_Learning::sampleAction(const FiniteState& state, FiniteAction& action)
{
    x = state.getStateN();
    updateQ();
    u = policy(x);

    action.setActionN(u);
}

void DoubleQ_Learning::step(const Reward& reward, const FiniteState& nextState,
                            FiniteAction& action)
{
    size_t xn = nextState.getStateN();
    unsigned int selectedQ;

    selectedQ = RandomGenerator::sampleUniformInt(0, 1);
    const arma::mat& Qxn = doubleQ.subcube(span(xn, xn), span::all, span(selectedQ, selectedQ));
    double qmax = Qxn.max();
    arma::uvec maxIndex = find(Qxn == qmax);
    unsigned int index = RandomGenerator::sampleUniformInt(0,
                         maxIndex.n_elem - 1);

    double delta = reward[0] +
                   task.gamma * doubleQ(xn, FiniteAction(maxIndex(index)), 1 - selectedQ) - doubleQ(x, u, selectedQ);
    doubleQ(x, u, selectedQ) = doubleQ(x, u, selectedQ) + alpha(x, u) * delta;

    // Update action and state
    updateQ();
    x = xn;
    u = policy(xn);

    action.setActionN(u);
}

void DoubleQ_Learning::endEpisode(const Reward& reward)
{
    unsigned int selectedQ = RandomGenerator::sampleUniformInt(0, 1);

    // In the last update, the Q of the absorbing state is forced to be 0
    double delta = reward[0] - doubleQ(x, u, selectedQ);
    doubleQ(x, u, selectedQ) = doubleQ(x, u, selectedQ) + alpha(x, u) * delta;

    updateQ();
}

void DoubleQ_Learning::init()
{
    FiniteTD::init();
    doubleQ.zeros(task.statesNumber, task.actionsNumber, 2);
}

inline void DoubleQ_Learning::updateQ()
{
    Q = (doubleQ.slice(0) + doubleQ.slice(1)) / 2;
}

DoubleQ_Learning::~DoubleQ_Learning()
{
}

}

