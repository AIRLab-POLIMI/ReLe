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

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_OFFGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_OFFGRADIENTCALCULATOR_H_

#include "rele/core/RewardTransformation.h"
#include "rele/core/Transition.h"
#include "rele/policy/Policy.h"

namespace ReLe
{

template<class ActionC, class StateC>
class OffGradientCalculator
{
public:
    OffGradientCalculator(RewardTransformation& rewardf, Dataset<ActionC,StateC>& data,
                          Policy<ActionC,StateC>& behaviour, DifferentiablePolicy<ActionC,StateC>& policy,
                          double gamma) :
        rewardf(rewardf), data(data), behaviour(behaviour), policy(policy), gamma(gamma)

    {

    }

    virtual arma::vec computeGradient(const arma::uvec& indices = arma::uvec()) = 0;

    virtual ~OffGradientCalculator()
    {

    }

protected:
    void computeEpisodeStatistics(Episode<ActionC,StateC>& episode, double& Rew, double& importanceWeights, arma::vec& sumGradLog)
    {
        //reset data
        Rew = 0;
        sumGradLog.zeros();

        double df = 1.0;
        double targetIW = 1.0;
        double behavoiourIW = 1.0;

        //iterate the episode
        for (int t = 0; t < episode.size(); ++t)
        {
            Transition<ActionC, StateC>& tr = episode[t];
            sumGradLog += this->policy.difflog(tr.x, tr.u);

            targetIW *= policy(tr.x, tr.u);
            behavoiourIW *= behaviour(tr.x, tr.u);

            Rew += df * arma::as_scalar(rewardf(tr.r));

            df *= gamma;
        }

        importanceWeights = targetIW / behavoiourIW;

    }

    inline unsigned int getEpisodeIndex(const arma::uvec& indexes, unsigned int i)
    {
        if(indexes.is_empty())
        {
            return i;
        }
        else
        {
            return indexes(i);
        }
    }

    inline unsigned int getEpisodesNumber(const arma::uvec& indexes)
    {
        if(indexes.is_empty())
        {
            return data.size();
        }
        else
        {
            return indexes.n_elem;
        }
    }

protected:
    RewardTransformation& rewardf;
    Dataset<ActionC,StateC>& data;
    Policy<ActionC,StateC>& behaviour;
    DifferentiablePolicy<ActionC,StateC>& policy;
    double gamma;
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_OFFGRADIENTCALCULATOR_H_ */

