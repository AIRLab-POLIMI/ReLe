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

#ifndef INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_EM_EPISODE_BASED_RWR_H_
#define INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_EM_EPISODE_BASED_RWR_H_

#include "rele/algorithms/policy_search/BlackBoxAlgorithm.h"
#include "rele/algorithms/policy_search/em/episode_based/EMOutputData.h"

namespace ReLe
{

template<class ActionC, class StateC>
class RWR: public BlackBoxAlgorithm<ActionC, StateC, EMOutputData>
{
    USE_BBA_MEMBERS(ActionC, StateC, EMOutputData)

public:
    RWR(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
        unsigned int nbEpisodes, unsigned int nbPolicies,
        RewardTransformation& reward_tr, double beta)
        : Base(dist, policy, nbEpisodes, nbPolicies, reward_tr), beta(beta)
    {

    }

    RWR(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
        unsigned int nbEpisodes, unsigned int nbPolicies,
        double beta, int reward_obj = 0)
        : Base(dist, policy, nbEpisodes, nbPolicies, reward_obj), beta(beta)
    {
    }


    virtual ~RWR()
    {

    }

protected:
    virtual void init() override
    {
        d.zeros(nbPoliciesToEvalMetap);
        theta.resize(policy.getParametersSize(), nbPoliciesToEvalMetap);
    }

    virtual void afterPolicyEstimate() override
    {
        d(polCount) = std::exp(beta*Jpol/nbEpisodesToEvalPolicy);
        theta.col(polCount) = policy.getParameters();
    }

    virtual void afterMetaParamsEstimate() override
    {
        dist.wmle(d, theta);
    }

private:
    arma::vec d;
    arma::mat theta;
    double beta;

};

}

#endif /* INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_EM_EPISODE_BASED_RWR_H_ */
