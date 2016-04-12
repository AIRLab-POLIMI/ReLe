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

#ifndef MLE_H_
#define MLE_H_

#include "rele/policy/utils/StatisticEstimation.h"

namespace ReLe
{

template<class ActionC, class StateC>
class MLE : public StatisticEstimation<ActionC, StateC>
{
protected:
    USING_STATISTIC_ESTIMATION_MEMBERS(ActionC, StateC)

public:
    MLE(DifferentiablePolicy<ActionC,StateC>& policy, Dataset<ActionC,StateC>& data)
        : StatisticEstimation<ActionC, StateC>(policy, data)
    {

    }

    virtual double objFunction(const arma::vec& x, arma::vec& dx) override
    {
        policy.setParameters(x);

        int nbEpisodes = data.size();
        double logLikelihood = 0.0;
        int counter = 0;
        for (int ep = 0; ep < nbEpisodes; ++ep)
        {
            int nbSteps = data[ep].size();
            for (int t = 0; t < nbSteps; ++t)
            {
                auto& tr = data[ep][t];

                // compute probability
                double prob = policy(tr.x,tr.u);
                prob = std::max(1e-8,prob);
                logLikelihood += std::log(prob);

                // compute gradient
                dx += policy.difflog(tr.x, tr.u);
            }
        }

        // compute average value
        logLikelihood /= nbEpisodes;

        return logLikelihood;
    }

    virtual ~MLE()
    {

    }
};

/**
 * Ridge Regularization
 */
template<class ActionC, class StateC>
class RidgeRegularizedMLE : public MLE<ActionC, StateC>
{

protected:
    USING_STATISTIC_ESTIMATION_MEMBERS(ActionC, StateC)

public:
    RidgeRegularizedMLE(DifferentiablePolicy<ActionC,StateC>& policy,
                        Dataset<ActionC,StateC>& ds,
                        double lambda = 1.0)
        : MLE<ActionC, StateC>(policy, ds), lambda(lambda)
    {
        assert(lambda >= 0.0);
    }

    virtual double objFunction(const arma::vec& x, arma::vec& dx) override
    {
        policy.setParameters(x);

        int nbEpisodes = data.size();
        double logLikelihood = 0.0;
        for (int ep = 0; ep < nbEpisodes; ++ep)
        {
            int nbSteps = data[ep].size();
            for (int t = 0; t < nbSteps; ++t)
            {
                auto& tr = data[ep][t];

                // compute probability
                double prob = policy(tr.x,tr.u);
                prob = std::max(1e-8,prob);
                logLikelihood += log(prob);

                // compute gradient
                dx += policy.difflog(tr.x, tr.u);
            }
        }

        // compute average value
        logLikelihood /= nbEpisodes;
        double l2normParams = arma::norm(x);
        logLikelihood -= lambda * l2normParams;
        arma::vec L2RegGradient = x / l2normParams;
        dx -= lambda * L2RegGradient;

        return logLikelihood;
    }

    virtual ~RidgeRegularizedMLE()
    {

    }

protected:
    double lambda;
};

} //end namespace
#endif //MLE_H_
