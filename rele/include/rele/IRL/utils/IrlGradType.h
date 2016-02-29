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

#ifndef INCLUDE_RELE_IRL_UTILS_IRLGRADTYPE_H_
#define INCLUDE_RELE_IRL_UTILS_IRLGRADTYPE_H_

#include <map>
#include <stdexcept>

namespace ReLe
{

enum IrlGrad
{
    REINFORCE, REINFORCE_BASELINE, GPOMDP, GPOMDP_BASELINE, ENAC, ENAC_BASELINE,
    NATURAL_REINFORCE, NATURAL_REINFORCE_BASELINE, NATURAL_GPOMDP, NATURAL_GPOMDP_BASELINE
};

class IrlGradUtils
{
public:
    static bool isValid(const std::string& type);
    static IrlGrad fromString(const std::string& type);
    static std::string toString(IrlGrad type);

private:
    static std::map<std::string, IrlGrad> initGradients();

private:
    static std::map<std::string, IrlGrad> gradients;


};

}



#endif /* INCLUDE_RELE_IRL_UTILS_IRLGRADTYPE_H_ */