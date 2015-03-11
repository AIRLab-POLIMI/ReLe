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

#include "q_policy/Boltzmann.h"
#include "RandomGenerator.h"

using namespace arma;
using namespace std;

namespace ReLe
{

Boltzmann::Boltzmann()
{
    Q = nullptr;
    nactions = 0;

    tau = 1;

}

unsigned int Boltzmann::operator()(size_t state)
{
    if (tau != 0)
    {
        vec&& p = computeProbabilities(state);
        return RandomGenerator::sampleDiscrete(p.begin(), p.end());
    }
    else
    {
        return RandomGenerator::sampleUniformInt(0, nactions);
    }
}

double Boltzmann::operator()(size_t state, unsigned int action)
{
    if (tau != 0)
    {
        vec&& p = computeProbabilities(state);
        return p(action);
    }
    else
    {
        return 1 / nactions;
    }
}

string Boltzmann::getPolicyHyperparameters()
{
    stringstream ss;

    ss << "tau: " << tau << endl;

    return ss.str();
}

Boltzmann::~Boltzmann()
{

}

vec Boltzmann::computeProbabilities(size_t state)
{
    const rowvec& Qx = Q->row(state);
    vec p(nactions);

    for (unsigned int i = 0; i < nactions; i++)
    {
        p[i] = std::exp(Qx(i)) / tau;
    }

    p = p / sum(p);

    return p;
}

BoltzmannApproximate::BoltzmannApproximate()
{
    Q = nullptr;
    nactions = 0;

    //Default parameter
    tau = 1;
}

unsigned int BoltzmannApproximate::operator()(const arma::vec& state)
{
    if (tau != 0)
    {
        vec&& p = computeProbabilities(state);
        return RandomGenerator::sampleDiscrete(p.begin(), p.end());
    }
    else
    {
        return RandomGenerator::sampleUniformInt(0, nactions);
    }
}

double BoltzmannApproximate::operator()(const arma::vec& state, unsigned int action)
{
    if (tau != 0)
    {
        vec&& p = computeProbabilities(state);
        return p(action);
    }
    else
    {
        return 1 / nactions;
    }
}

string BoltzmannApproximate::getPolicyHyperparameters()
{
    stringstream ss;

    ss << "tau: " << tau << endl;

    return ss.str();
}


BoltzmannApproximate::~BoltzmannApproximate()
{

}

arma::vec BoltzmannApproximate::computeProbabilities(const arma::vec& state)
{
    Regressor& Q = *this->Q;

    unsigned int nstates = state.size();
    vec regInput(nstates + 1);

    regInput.subvec(0, nstates - 1) = state;

    vec p(nactions);

    for (unsigned int i = 0; i < nactions; i++)
    {
        regInput[nstates] = i;
        p[i] = std::exp(Q(regInput)[0]) / tau;
    }

    p = p / sum(p);
    return p;

}

}
