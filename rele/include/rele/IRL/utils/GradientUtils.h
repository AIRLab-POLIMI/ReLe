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

#ifndef INCLUDE_RELE_IRL_UTILS_GRADIENTUTILS_H_
#define INCLUDE_RELE_IRL_UTILS_GRADIENTUTILS_H_

#include "rele/IRL/utils/gradient/ReinforceGradientCalculator.h"
#include "rele/IRL/utils/gradient/GPOMDPGradientCalculator.h"
/*#include "rele/IRL/utils/gradient/ENACGradientCalculator.h"*/
#include "rele/IRL/utils/gradient/NaturalGradientCalculator.h"
#include "rele/IRL/utils/IrlGradType.h"

namespace ReLe
{

template<class ActionC, class StateC>
class GradientCalculatorFactory
{

public:
    static GradientCalculator<ActionC, StateC>* build(IrlGrad type,
            Features& phi,
            Dataset<ActionC,StateC>& data,
            DifferentiablePolicy<ActionC,StateC>& policy,
            double gamma)
    {

        switch(type)
        {
        case REINFORCE:
            return new ReinforceGradientCalculator<ActionC, StateC>(phi, data, policy, gamma);

        case REINFORCE_BASELINE:
            return new ReinforceBaseGradientCalculator<ActionC, StateC>(phi, data, policy, gamma);

        case GPOMDP:
            return new GPOMDPGradientCalculator<ActionC, StateC>(phi, data, policy, gamma);

        case GPOMDP_BASELINE:
            return new GPOMDPBaseGradientCalculator<ActionC, StateC>(phi, data, policy, gamma);

        /*case ENAC:
            return new ENACGradientCalculator<ActionC, StateC>(basis, data, policy, gamma);

        case ENAC_BASELINE:
            return new ENACGradientCalculator<ActionC, StateC>(basis, data, policy, gamma);*/

        default:
            return buildNatural(type, phi, data, policy, gamma);

        }
    }

private:
    static GradientCalculator<ActionC, StateC>* buildNatural(IrlGrad type,
            Features& phi,
            Dataset<ActionC,StateC>& data,
            DifferentiablePolicy<ActionC,StateC>& policy,
            double gamma)
    {
        switch(type)
        {
        case NATURAL_REINFORCE:
            return new NaturalGradientCalculator<ActionC, StateC, ReinforceGradientCalculator<ActionC, StateC>>(phi, data, policy, gamma);

        case NATURAL_REINFORCE_BASELINE:
            return new NaturalGradientCalculator<ActionC, StateC, ReinforceBaseGradientCalculator<ActionC, StateC>>(phi, data, policy, gamma);

        case NATURAL_GPOMDP:
            return new NaturalGradientCalculator<ActionC, StateC, GPOMDPGradientCalculator<ActionC, StateC>>(phi, data, policy, gamma);

        case NATURAL_GPOMDP_BASELINE:
            return new NaturalGradientCalculator<ActionC, StateC, GPOMDPBaseGradientCalculator<ActionC, StateC>>(phi, data, policy, gamma);

        default:
            return nullptr;
        }
    }


};

}



#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENTUTILS_H_ */
