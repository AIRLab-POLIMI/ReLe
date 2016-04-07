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

#include "rele/policy/nonparametric/TabularPolicy.h"
#include "rele/utils/RandomGenerator.h"

#include <sstream>

namespace ReLe
{

TabularPolicy::updater::updater(arma::subview_row<double>&& row) :
    row(row)
{
    nactions = row.n_elem;
    currentIndex = 0;
}

void TabularPolicy::updater::operator<<(double weight)
{
    row(currentIndex) = weight;
    currentIndex++;
}

void TabularPolicy::updater::normalize()
{
    double normalization = sum(row);
    row /= normalization;
}

unsigned int TabularPolicy::operator()(const size_t& state)
{
    arma::rowvec&& row = pi.row(state);
    return RandomGenerator::sampleDiscrete(row.begin(), row.end());
}

double TabularPolicy::operator()(const size_t& state, const unsigned int& action)
{
    return pi(state, action);
}

std::string TabularPolicy::printPolicy()
{
    //TODO [MINIOR] choose policy format
    std::stringstream ss;
    ss << "- Policy" << std::endl;
    for (unsigned int i = 0; i < pi.n_rows; i++)
    {
        arma::uword policy;
        pi.row(i).max(policy);
        ss << "policy(" << i << ") = " << policy << std::endl;
    }

    return ss.str();
}

}
