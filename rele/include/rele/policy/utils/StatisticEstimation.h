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

#ifndef INCLUDE_RELE_POLICY_UTILS_STATISTICESTIMATION_H_
#define INCLUDE_RELE_POLICY_UTILS_STATISTICESTIMATION_H_

#include "rele/policy/Policy.h"
#include "rele/core/Transition.h"
#include "rele/optimization/Optimization.h"

#include <nlopt.hpp>
#include <cassert>

namespace ReLe
{

/*!
 * This interface represent a basic statistic estimation algorith, i.e.
 * an algorithm that uses statistic inference for learning a policy from
 * a given dataset.
 */
template<class ActionC, class StateC>
class StatisticEstimation
{
public:
    StatisticEstimation(DifferentiablePolicy<ActionC,StateC>& policy,
                        const Dataset<ActionC,StateC>& data) :
        policy(policy), data(data)
    {
    }

    virtual ~StatisticEstimation()
    {

    }

    virtual double compute(arma::vec starting = arma::vec(),
                           unsigned int maxFunEvals = 0)
    {
        int dp = policy.getParametersSize();
        assert(dp > 0);

        if (starting.n_elem == 0)
        {
            starting.zeros(dp);
        }
        else
        {
            assert(dp == starting.n_elem);
        }

        if (maxFunEvals == 0)
            maxFunEvals = std::min(30*dp, 600);

        nlopt::opt optimizator;
        optimizator = nlopt::opt(nlopt::algorithm::LD_SLSQP, dp);
        optimizator.set_max_objective(Optimization::objFunctionWrapper<StatisticEstimation, false>, this);
        optimizator.set_ftol_rel(1e-10);
        optimizator.set_ftol_abs(1e-10);
        optimizator.set_maxeval(maxFunEvals);

        //optimize the function
        std::vector<double> parameters(dp, 0.0);
        for (int i = 0; i < dp; ++i)
            parameters[i] = starting[i];
        double minf;

        if (optimizator.optimize(parameters, minf) < 0)
        {
            throw std::runtime_error("nlopt failed!");
        }
        else
        {
            arma::vec finalP(dp);
            for(int i = 0; i < dp; ++i)
            {
                finalP(i) = parameters[i];
            }

            policy.setParameters(finalP);

            return minf;
        }
    }

    virtual double objFunction(const arma::vec& x, arma::vec& dx) = 0;

protected:
    DifferentiablePolicy<ActionC,StateC>& policy;
    const Dataset<ActionC,StateC>& data;


};

#define USING_STATISTIC_ESTIMATION_MEMBERS(ActionC, StateC) \
		typedef StatisticEstimation<ActionC, StateC> Base; \
		using Base::policy; \
		using Base::data; \
 
}

#endif /* INCLUDE_RELE_POLICY_UTILS_STATISTICESTIMATION_H_ */
