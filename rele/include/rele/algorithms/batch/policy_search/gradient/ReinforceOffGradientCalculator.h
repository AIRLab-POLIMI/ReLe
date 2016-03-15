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

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_REINFORCEOFFGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_REINFORCEOFFGRADIENTCALCULATOR_H_

#include "rele/algorithms/batch/policy_search/gradient/OffGradientCalculator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class ReinforceOffGradientCalculator : public OffGradientCalculator<ActionC, StateC>
{
public:
    ReinforceOffGradientCalculator(RewardTransformation& rewardf, Dataset<ActionC,StateC>& data,
                                   Policy<ActionC,StateC>& behaviour, DifferentiablePolicy<ActionC,StateC>& policy,
                                   double gamma)
        : OffGradientCalculator<ActionC, StateC>(rewardf, data, behaviour, policy, gamma)

    {

    }

    virtual arma::vec computeGradient(const arma::uvec& indexes = arma::uvec()) override
    {
        unsigned int episodeN = this->getEpisodesNumber(indexes);
        unsigned int parametersN = this->policy.getParametersSize();

        double importanceWeightsSum = 0;

        arma::vec gradient(parametersN, arma::fill::zeros);

        //iterate episodes
        for (int i = 0; i < episodeN; ++i)
        {
            double Rew;
            double iw;
            arma::vec sumGradLog(parametersN);

            unsigned int ep = this->getEpisodeIndex(indexes, i);
            this->computeEpisodeStatistics(this->data[ep], Rew, iw, sumGradLog);

            gradient += sumGradLog * Rew * iw;

            importanceWeightsSum += iw;
        }

        // Compute mean values
        gradient /= importanceWeightsSum;

        return gradient;

    }

    virtual ~ReinforceOffGradientCalculator()
    {

    }

};

template<class ActionC, class StateC>
class ReinforceBaseOffGradientCalculator : public OffGradientCalculator<ActionC, StateC>
{
public:
    ReinforceBaseOffGradientCalculator(RewardTransformation& rewardf, Dataset<ActionC,StateC>& data,
                                       Policy<ActionC,StateC>& behaviour, DifferentiablePolicy<ActionC,StateC>& policy,
                                       double gamma)
        : OffGradientCalculator<ActionC, StateC>(rewardf, data, behaviour, policy, gamma)

    {

    }

    virtual arma::vec computeGradient(const arma::uvec& indexes = arma::uvec()) override
    {
        int parametersN = this->policy.getParametersSize();
        int episodeN = this->getEpisodesNumber(indexes);

        // Reset computed results
        arma::vec gradient(parametersN, arma::fill::zeros);

        // gradient basics
        arma::vec Rew_ep(episodeN);
        arma::vec iw_ep(episodeN);
        arma::mat sumGradLog_ep(parametersN, episodeN);

        // baselines
        arma::vec baseline_den(parametersN, arma::fill::zeros);
        arma::vec baseline_num(parametersN, arma::fill::zeros);

        for (int i = 0; i < episodeN; ++i)
        {
            // compute basic elements
            double Rew;
            double iw;
            arma::vec sumGradLog(parametersN);
            unsigned int ep = this->getEpisodeIndex(indexes, i);
            this->computeEpisodeStatistics(this->data[ep], Rew, iw, sumGradLog);

            // store them
            Rew_ep(i) = Rew;
            iw_ep(i) = iw;
            sumGradLog_ep.col(i) = sumGradLog;

            // compute baseline num and den
            arma::vec sumGradLog2 = sumGradLog % sumGradLog;
            baseline_den += sumGradLog2 * iw * iw;
            baseline_num += sumGradLog2 * Rew * iw * iw;
        }

        // compute the gradients
        arma::vec baseline = baseline_num / baseline_den;
        baseline(arma::find_nonfinite(baseline)).zeros();

        for (int ep = 0; ep < episodeN; ep++)
        {
            gradient += (Rew_ep(ep) - baseline) % sumGradLog_ep.col(ep) * iw_ep(ep);
        }

        // compute mean values
        gradient /= arma::sum(iw_ep);

        return gradient;

    }

    virtual ~ReinforceBaseOffGradientCalculator()
    {

    }

};


}




#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_REINFORCEOFFGRADIENTCALCULATOR_H_ */
