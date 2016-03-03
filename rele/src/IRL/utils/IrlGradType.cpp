/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#include "rele/IRL/utils/IrlGradType.h"

namespace ReLe
{

bool IrlGradUtils::isValid(const std::string& type)
{
    return gradients.count(type) != 0;
}

IrlGrad IrlGradUtils::fromString(const std::string& type)
{
    if(gradients.count(type))
    {
        return gradients[type];
    }
    else
        throw std::runtime_error("Unknown type");
}

std::string IrlGradUtils::toString(IrlGrad type)
{
    switch(type)
    {
    case IrlGrad::REINFORCE:
        return "REINFORCE";
    case IrlGrad::REINFORCE_BASELINE:
        return "REINFORCE_BASELINE";
    case IrlGrad::GPOMDP:
        return "GPOMDP";
    case IrlGrad::GPOMDP_BASELINE:
        return "GPOMDP_BASELINE";
    case IrlGrad::ENAC:
        return "ENAC";
    case IrlGrad::ENAC_BASELINE:
        return "ENAC_BASELINE";
    case IrlGrad::NATURAL_REINFORCE:
        return "NATURAL_REINFORCE";
    case IrlGrad::NATURAL_REINFORCE_BASELINE:
        return "NATURAL_REINFORCE_BASELINE";
    case IrlGrad::NATURAL_GPOMDP:
        return "NATURAL_GPOMDP";
    case IrlGrad::NATURAL_GPOMDP_BASELINE:
        return "NATURAL_GPOMDP_BASELINE";
    default:
        throw std::runtime_error("Unknown type");
    }
}


std::map<std::string, IrlGrad> IrlGradUtils::initGradients()
{
    std::map<std::string, IrlGrad> map;

    map["REINFORCE"] = IrlGrad::REINFORCE;
    map["REINFORCE_BASELINE"] = IrlGrad::REINFORCE_BASELINE;
    map["GPOMDP"] = IrlGrad::GPOMDP;
    map["GPOMDP_BASELINE"] = IrlGrad::GPOMDP_BASELINE;
    map["ENAC"] = IrlGrad::ENAC;
    map["ENAC_BASELINE"] = IrlGrad::ENAC_BASELINE;
    map["NATURAL_REINFORCE"] = IrlGrad::NATURAL_REINFORCE;
    map["NATURAL_REINFORCE_BASELINE"] = IrlGrad::NATURAL_REINFORCE_BASELINE;
    map["NATURAL_GPOMDP"] = IrlGrad::NATURAL_GPOMDP;
    map["NATURAL_GPOMDP_BASELINE"] = IrlGrad::NATURAL_GPOMDP_BASELINE;

    return map;
}

std::map<std::string, IrlGrad> IrlGradUtils::gradients = IrlGradUtils::initGradients();

}

