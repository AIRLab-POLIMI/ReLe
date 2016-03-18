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

#ifndef INCLUDE_RELE_IRL_UTILS_STEPBASEDGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_STEPBASEDGRADIENTCALCULATOR_H_

#include <cassert>

#include "rele/policy/Policy.h"
#include "rele/core/Transition.h"
#include "rele/approximators/Features.h"

#include "rele/IRL/utils/gradient/GradientCalculator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class StepBasedGradientCalculator : public GradientCalculator<ActionC, StateC>
{
public:
    StepBasedGradientCalculator(Features& phi,
                                Dataset<ActionC,StateC>& data,
                                DifferentiablePolicy<ActionC,StateC>& policy,
                                double gamma):
        GradientCalculator<ActionC, StateC>(policy.getParametersSize(), phi.rows(), gamma),
        phi(phi), data(data), policy(policy)
    {
        assert(phi.cols() == 1);
    }

    virtual ~StepBasedGradientCalculator()
    {

    }


protected:

    arma::vec computeSumGradLog(Episode<ActionC, StateC>& episode)
    {
        int dp  = policy.getParametersSize();
        int nbSteps = episode.size();
        arma::vec sumGradLog(dp, arma::fill::zeros);

        //iterate the episode
        for (int t = 0; t < nbSteps; ++t)
        {
            Transition<ActionC, StateC>& tr = episode[t];
            sumGradLog += policy.difflog(tr.x, tr.u);
        }

        return sumGradLog;
    }




protected:
    Features& phi;
    Dataset<ActionC,StateC>& data;
    DifferentiablePolicy<ActionC,StateC>& policy;


};


}

#define USING_STEP_BASED_CALCULATORS_MEMBERS(ActionC, StateC) \
	typedef StepBasedGradientCalculator<ActionC, StateC> Base; \
	using Base::gamma; \
	using Base::phi; \
	using Base::data; \
	using Base::policy;



#endif /* INCLUDE_RELE_IRL_UTILS_STEPBASEDGRADIENTCALCULATOR_H_ */
