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

#include "td/Q-Learning.h"
#include "RandomGenerator.h"

using namespace std;
using namespace arma;

namespace ReLe
{

Q_Learning::Q_Learning(size_t statesN, size_t actionN) :
    Q(statesN, actionN, fill::zeros)
{
    x = 0;
    u = 0;

    //Default algorithm parameters
    alpha = 0.2;
    eps = 0.15;
}

void Q_Learning::initEpisode(const FiniteState& state,  FiniteAction& action)
{
    sampleAction(state, action);
}

void Q_Learning::sampleAction(const FiniteState& state, FiniteAction& action)
{
    x = state.getStateN();
    u = policy(x);

    action.setActionN(u);
}

void Q_Learning::step(const Reward& reward, const FiniteState& nextState, FiniteAction& action)
{
    size_t xn = nextState.getStateN();
    double r = reward[0];
    double maxQxn;

    const rowvec& Qxn = Q.row(xn);
    maxQxn = Qxn.max();

    double delta = r + task.gamma * maxQxn - Q(x, u);
    Q(x, u) = Q(x, u) + alpha * delta;

    //update action and state
    x = xn;
    u = policy(xn);

    //set next action
    action.setActionN(u);
}

void Q_Learning::endEpisode()
{
    //print statistics
    printStatistics();
}

void Q_Learning::endEpisode(const Reward& reward)
{
    //Last update
    double r = reward[0];
    double delta = r - Q(x, u);
    Q(x, u) = Q(x, u) + alpha * delta;

    //print statistics
    printStatistics();
}

Q_Learning::~Q_Learning()
{

}

unsigned int Q_Learning::policy(size_t x)
{
    unsigned int un;

    const rowvec& Qx = Q.row(x);

    /*epsilon--greedy policy*/
    if (RandomGenerator::sampleEvent(this->eps))
        un = RandomGenerator::sampleUniformInt(0, Q.n_cols - 1);
    else
        Qx.max(un);

    return un;
}

void Q_Learning::printStatistics()
{
    //TODO dentro la classe o altrove???
    cout << endl << endl << "### Q-Learning ###";

    cout << endl << endl << "--- Parameters --"
         << endl << endl;
    cout << "gamma: " << gamma << endl;
    cout << "alpha: " << alpha << endl;
    cout << "eps: " << eps << endl;

    cout << endl << endl << "--- Learning results ---"
         << endl << endl;

    cout << "- Action-value function" << endl;
    for (unsigned int i = 0; i < Q.n_rows; i++)
        for (unsigned int j = 0; j < Q.n_cols; j++)
        {
            cout << "Q(" << i << ", " << j << ") = " << Q(i, j) << endl;
        }
    cout << "- Policy" << endl;
    for (unsigned int i = 0; i < Q.n_rows; i++)
    {
        unsigned int policy;
        Q.row(i).max(policy);
        cout << "policy(" << i << ") = " << policy << endl;
    }

}

}

