/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#include "td/TD.h"
#include "RandomGenerator.h"

using namespace std;
using namespace arma;

namespace ReLe
{

FiniteTD::FiniteTD()
    : e_policy(&Q)
{
    x = 0;
    u = 0;

    //Default algorithm parameters
    alpha = 0.2;
    e_policy.setEpsilon(0.15);
}

void FiniteTD::init()
{
    Q.zeros(task.finiteStateDim, task.finiteActionDim);
}

unsigned int FiniteTD::policy(std::size_t x)
{
    return e_policy(x);
}

void FiniteTD::printStatistics()
{
    cout << endl << endl << "--- Parameters --" << endl << endl;
    cout << "gamma: " << gamma << endl;
    cout << "alpha: " << alpha << endl;
    cout << "eps: " << e_policy.getEpsilon() << endl;

    cout << endl << endl << "--- Learning results ---" << endl << endl;

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

LinearTD::LinearTD(LinearApproximator &la) :
    Q(la), x(task.continuosStateDim), e_policy(&Q, task.finiteActionDim)
{
    u = 0;

    //Default parameters
    alpha = 0.2;
    e_policy.setEpsilon(0.15);
}

unsigned int LinearTD::policy(DenseState state)
{
    return e_policy(state);
}

void LinearTD::printStatistics()
{
    cout << endl << endl << "--- Parameters --" << endl << endl;
    cout << "gamma: " << gamma << endl;
    cout << "alpha: " << alpha << endl;
    cout << "eps: " << e_policy.getEpsilon() << endl;

    cout << endl << endl << "--- Learning results ---" << endl << endl;

    cout << "- Action-value function" << endl;
    cout << Q.getParameters().t() << endl;
//    cout << "- Policy" << endl;
}

}
