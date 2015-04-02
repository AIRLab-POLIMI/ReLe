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

#include "BasicDynamicProgramming.h"

using namespace std;
using namespace arma;

namespace ReLe
{

DynamicProgrammingAlgorithm::DynamicProgrammingAlgorithm(FiniteMDP& mdp) : mdp(mdp), P(mdp.P), R(mdp.R)
{
    const EnvirormentSettings& settings = mdp.getSettings();
    stateN = settings.finiteStateDim;
    actionN = settings.finiteActionDim;
    gamma = settings.gamma;
}

ValueIteration::ValueIteration(FiniteMDP& mdp, double eps) : DynamicProgrammingAlgorithm(mdp), eps(eps)
{
    V.zeros(stateN);
}

void ValueIteration::solve()
{
    do
    {
        Vold = V;

        for(size_t s = 0; s < stateN; s++)
        {
            double vmax = -numeric_limits<double>::infinity();

            for(unsigned int a = 0; a < actionN; a++)
            {
                arma::vec Psa = P.tube(a, s);
                arma::vec Rsa = R.tube(a, s);
                arma::vec va = Psa.t()*(Rsa+gamma*Vold);
                vmax = max(va[0], vmax);
            }

            V[s] = vmax;
        }
    }
    while(norm(V-Vold) < eps);

}

Dataset<FiniteAction, FiniteState> ValueIteration::test()
{

}

string ValueIteration::printPolicy()
{

}

ValueIteration::~ValueIteration()
{

}


}
