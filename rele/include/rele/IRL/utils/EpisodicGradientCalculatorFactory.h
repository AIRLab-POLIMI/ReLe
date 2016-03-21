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

#ifndef INCLUDE_RELE_IRL_UTILS_EPISODICGRADIENTCALCULATORFACTORY_H_
#define INCLUDE_RELE_IRL_UTILS_EPISODICGRADIENTCALCULATORFACTORY_H_

#include "rele/IRL/utils/gradient/episodic/PGPEGradientCalculator.h"
#include "rele/IRL/utils/IrlGradType.h"

namespace ReLe
{

template<class ActionC, class StateC>
class EpisodicGradientCalculatorFactory
{

public:
    static GradientCalculator<ActionC, StateC>* build(IrlEpGrad type,
            const arma::mat& theta,
            const arma::mat& phi,
            DifferentiableDistribution& dist,
            double gamma)
    {

        switch(type)
        {
        case IrlEpGrad::PGPE:
            return new PGPEGradientCalculator<ActionC, StateC>(theta, phi, dist, gamma);
        case IrlEpGrad::PGPE_BASELINE:
            return new PGPEBaseGradientCalculator<ActionC, StateC>(theta, phi, dist, gamma);
        }
    }




};

}



#endif /* INCLUDE_RELE_IRL_UTILS_EPISODICGRADIENTCALCULATORFACTORY_H_ */
