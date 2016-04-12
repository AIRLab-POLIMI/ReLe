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

#ifndef INCLUDE_RELE_POLICY_UTILS_MAP_H_
#define INCLUDE_RELE_POLICY_UTILS_MAP_H_

#include "rele/policy/utils/StatisticEstimation.h"

namespace ReLe
{

/*!
 * This class can be used to compute the Maximum A Posteriori estimate
 * for a generic parametric stochastic policy.
 */
template<class ActionC, class StateC>
class MAP : public StatisticEstimation<ActionC, StateC>
{
protected:
    USING_STATISTIC_ESTIMATION_MEMBERS(ActionC, StateC)

public:
    MAP(DifferentiablePolicy<ActionC,StateC>& policy,
        const DifferentiableDistribution& prior,
        const Dataset<ActionC,StateC>& data)
        : StatisticEstimation<ActionC, StateC>(policy, data), prior(prior)
    {

    }

    virtual double objFunction(const arma::vec& x, arma::vec& dx) override
    {
        policy.setParameters(x);

        unsigned int episodeN = data.size();
        unsigned int dp = policy.getParametersSize();
        double logLikelihood = 0.0;

        for (int ep = 0; ep < episodeN; ++ep)
        {
            int nbSteps = data[ep].size();
            for (int t = 0; t < nbSteps; ++t)
            {
                auto& tr = data[ep][t];

                // compute likelihood
                double likelihood = policy(tr.x,tr.u);
                likelihood = std::max(1e-300,likelihood);
                logLikelihood += log(likelihood);

                // compute gradient
                dx += policy.difflog(tr.x, tr.u);
            }
        }

        // compute average value
        logLikelihood /= episodeN;
        dx /= episodeN;

        //compute prior
        double priorP = prior(x);
        double logPrior = std::log(priorP);
        dx += prior.pointDifflog(x);

        return logLikelihood + logPrior;
    }

    virtual ~MAP()
    {

    }

protected:
    const DifferentiableDistribution& prior;

};

}


#endif /* INCLUDE_RELE_POLICY_UTILS_MAP_H_ */
