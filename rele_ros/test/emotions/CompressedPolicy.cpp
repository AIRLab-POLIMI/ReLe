/*
 * rele_ros,
 *
 *
 * Copyright (C) 2017 Davide Tateo
 * Versione 1.0
 *
 * This file is part of rele_ros.
 *
 * rele_ros is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rele_ros is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with rele_ros.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "CompressedPolicy.h"

using namespace ReLe;

namespace ReLe_ROS
{

CompressedPolicy::CompressedPolicy(Features& phi, Autoencoder& decoder) :
    approximator(phi), decoder(decoder)
{
}

CompressedPolicy::~CompressedPolicy()
{
}

std::string CompressedPolicy::getPolicyName()
{
    return std::string("CompressedPolicy");
}

std::string CompressedPolicy::printPolicy()
{
    /*std::stringstream ss;
    ss << approximator.getParameters().t();
    return ss.str();*/
    //TODO [IMPLEMENT]
    return "";
}

arma::vec CompressedPolicy::operator()(const arma::vec& state)
{
    return approximator(state);
}

double CompressedPolicy::operator()(const arma::vec& state,
                                    const arma::vec& action)
{
    arma::vec output = approximator(state);

    DenseAction a(output);

    if (a.isAlmostEqual(action))
    {
        return 1.0;
    }
    return 0.0;
}

CompressedPolicy* CompressedPolicy::clone()
{
    return new CompressedPolicy(*this);
}

arma::vec CompressedPolicy::getParameters() const
{
    return approximator.getParameters();
}
const unsigned int CompressedPolicy::getParametersSize() const
{
    return approximator.getParametersSize();
}
void CompressedPolicy::setParameters(const arma::vec& w)
{
    arma::vec wDecoded = decoder.decode(w);
    approximator.setParameters(wDecoded);
}

}
