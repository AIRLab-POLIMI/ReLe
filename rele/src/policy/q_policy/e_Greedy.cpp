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

#include "q_policy/e_Greedy.h"
#include "RandomGenerator.h"

#include <sstream>

using namespace arma;
using namespace std;

namespace ReLe
{

e_Greedy::e_Greedy()
{
    Q = NULL;
    eps = 0.15;
    nactions = 0;
}

e_Greedy::~e_Greedy()
{

}

int e_Greedy::operator()(const int state)
{
    unsigned int un;

    const rowvec& Qx = Q->row(state);

    /*epsilon--greedy policy*/
    if (RandomGenerator::sampleEvent(this->eps))
        un = RandomGenerator::sampleUniformInt(0, Q->n_cols - 1);
    else
    {
        double qmax = Qx.max();
        uvec maxIndex = find(Qx == qmax);

        unsigned int index = RandomGenerator::sampleUniformInt(0,
                             maxIndex.n_elem - 1);
        un = maxIndex[index];
    }

    return un;
}

double e_Greedy::operator()(const int state, const int action)
{
    const rowvec& Qx = Q->row(state);
    double qmax = Qx.max();
    uvec maxIndex = find(Qx == qmax);

    bool found = false;
    for (unsigned int i = 0; i < maxIndex.n_elem && !found; ++i)
    {
        if (maxIndex[i] == action)
            found = true;
    }
    if (found)
    {
        return 1.0 - eps + eps / Q->n_cols;
    }
    return eps / Q->n_cols;
}

string e_Greedy::getPolicyHyperparameters()
{
    stringstream ss;
    ss << "eps: " << eps << endl;

    return ss.str();
}

e_GreedyApproximate::e_GreedyApproximate()
{
    Q = NULL;
    nactions = 0;
    eps = 0.15;
}

e_GreedyApproximate::~e_GreedyApproximate()
{
}

int e_GreedyApproximate::operator()(const arma::vec& state)
{
    unsigned int un;

    /*epsilon--greedy policy*/
    if (RandomGenerator::sampleEvent(this->eps))
        un = RandomGenerator::sampleUniformInt(0, nactions - 1);
    else
    {
        Regressor& Qref = *Q;

        unsigned int nstates = state.size();
        vec regInput(nstates + 1);

        regInput.subvec(0, nstates - 1) = state;
        regInput[nstates] = 0;

        vec&& qvalue0 = Qref(regInput);
        double qmax = qvalue0[0];
        un = 0;
        for (unsigned int i = 1; i < nactions; ++i)
        {
            regInput[nstates] = i;
            vec&& qvalue = Qref(regInput);
            if (qmax < qvalue[0])
            {
                qmax = qvalue[0];
                un = i;
            }
        }
    }

    return un;
}

double e_GreedyApproximate::operator()(const arma::vec& state, const int action)
{
    //TODO implement
    assert(false);
    return 0.0;
}

string e_GreedyApproximate::getPolicyHyperparameters()
{
    stringstream ss;
    ss << "eps: " << eps << endl;
    return ss.str();
}

}
