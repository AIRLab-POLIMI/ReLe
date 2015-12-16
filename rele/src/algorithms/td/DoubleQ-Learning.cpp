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

#include "td/DoubleQ-Learning.h"
#include "RandomGenerator.h"

using namespace std;
using namespace arma;


namespace ReLe
{

DoubleQ_Learning::DoubleQ_Learning(ActionValuePolicy<FiniteState>& policy) :
    Q_Learning(policy)
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
    double r = reward[0];
    unsigned int selectedQ;

    selectedQ = RandomGenerator::sampleUniformInt(0, 1);
    if(selectedQ == 0)
    {
        const rowvec& Qxn = QA.row(xn);
        double qmax = Qxn.max();
        uvec maxIndex = find(Qxn == qmax);
        unsigned int index = RandomGenerator::sampleUniformInt(0,
                             maxIndex.n_elem - 1);

        double delta = r + task.gamma * QB(xn, FiniteAction(index)) - QA(x, u);
        QA(x, u) = QA(x, u) + alpha * delta;
    }
    else
    {
        const rowvec& Qxn = QB.row(xn);
        double qmax = Qxn.max();
        uvec maxIndex = find(Qxn == qmax);
        unsigned int index = RandomGenerator::sampleUniformInt(0,
                             maxIndex.n_elem - 1);

        double delta = r + task.gamma * QA(xn, FiniteAction(index)) - QB(x, u);
        QB(x, u) = QB(x, u) + alpha * delta;
    }

    //update action and state
    x = xn;
    updateQ();
    u = policy(x);

    action.setActionN(u);
}

void DoubleQ_Learning::endEpisode(const Reward& reward)
{
    //Last update
    double r = reward[0];

    unsigned int selectedQ = RandomGenerator::sampleUniformInt(0, 1);
    if(selectedQ == 0)
    {
        double delta = r - QA(x, u);
        QA(x, u) = QA(x, u) + alpha * delta;
    }
    else
    {
        double delta = r - QB(x, u);
        QB(x, u) = QB(x, u) + alpha * delta;
    }

    updateQ();
}

void DoubleQ_Learning::updateQ()
{
    Q = (QA + QB) / 2;
}

void DoubleQ_Learning::init()
{
    FiniteTD::init();
    QA.zeros(task.finiteStateDim, task.finiteActionDim);
    QB.zeros(task.finiteStateDim, task.finiteActionDim);
}

DoubleQ_Learning::~DoubleQ_Learning()
{
}

}

