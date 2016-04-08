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

#include "rele/solvers/dynamic_programming/BasicDynamicProgramming.h"
#include "rele/utils/CSV.h"
#include "rele/core/Core.h"

using namespace std;
using namespace arma;

namespace ReLe
{

DynamicProgrammingAlgorithm::DynamicProgrammingAlgorithm(FiniteMDP& mdp) : mdp(mdp), P(mdp.P), R(mdp.R)
{
    const EnvironmentSettings& settings = mdp.getSettings();
    stateN = settings.statesNumber;
    actionN = settings.actionsNumber;
    gamma = settings.gamma;
    pi.init(stateN);
}

Policy<FiniteAction, FiniteState>& DynamicProgrammingAlgorithm::getPolicy()
{
    return pi;
}

Dataset<FiniteAction, FiniteState> DynamicProgrammingAlgorithm::test()
{
    return Solver<FiniteAction, FiniteState>::test(mdp, pi);
}

ValueIteration::ValueIteration(FiniteMDP& mdp, double eps) : DynamicProgrammingAlgorithm(mdp), eps(eps)
{
    V.zeros(stateN);
}

void ValueIteration::solve()
{
    computeValueFunction();

    computePolicy();
}

ValueIteration::~ValueIteration()
{

}


void ValueIteration::computeValueFunction()
{
    do
    {
        Vold = V;

        for (size_t s = 0; s < stateN; s++)
        {
            double vmax = -numeric_limits<double>::infinity();
            for (unsigned int a = 0; a < actionN; a++)
            {
                arma::vec Psa = P.tube(a, s);
                arma::vec Rsa = R.tube(a, s);
                arma::vec va = Psa.t() * (Rsa + gamma * Vold);
                vmax = max(va[0], vmax);
            }

            V[s] = vmax;
        }
    }
    while (norm(V - Vold) > eps);
}

void ValueIteration::computePolicy()
{
    for (size_t s = 0; s < stateN; s++)
    {
        double vmax = -numeric_limits<double>::infinity();
        for (unsigned int a = 0; a < actionN; a++)
        {
            arma::vec Psa = P.tube(a, s);
            arma::vec Rsa = R.tube(a, s);
            arma::vec va = Psa.t() * (Rsa + gamma * V);
            if (va[0] > vmax)
            {
                vmax = va[0];
                pi.update(s, a);
            }
        }
    }
}

PolicyIteration::PolicyIteration(FiniteMDP& mdp) : DynamicProgrammingAlgorithm(mdp)
{
    changed = false;
}

void PolicyIteration::solve()
{
    do
    {
        computeValueFunction();
        computePolicy();
    }
    while(changed);
}

PolicyIteration::~PolicyIteration()
{

}

void PolicyIteration::computeValueFunction()
{
    mat Ppi(stateN, stateN);
    vec Rpi(stateN);
    mat I(stateN, stateN, fill::eye);

    for(size_t s = 0; s < stateN; s++)
    {
        unsigned int a = pi(s);
        vec PpiS = P.tube(a, s);
        vec RpiS = R.tube(a, s);

        Ppi.row(s) = PpiS.t();
        Rpi.row(s) = PpiS.t()*RpiS;
    }

    V = inv(I - gamma*Ppi)*Rpi;
}


void PolicyIteration::computePolicy()
{
    changed = false;

    for (size_t s = 0; s < stateN; s++)
    {
        double vmax = V[s];
        for (unsigned int a = 0; a < actionN; a++)
        {
            if (a != pi(s))
            {
                arma::vec Psa = P.tube(a, s);
                arma::vec Rsa = R.tube(a, s);
                arma::vec va = Psa.t() * (Rsa + gamma * V);
                if (va[0] > vmax)
                {
                    pi.update(s,a);
                    vmax = va[0];
                    changed = true;
                }
            }
        }
    }

}

}
