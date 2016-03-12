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

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_GPOMDPOFFGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_GPOMDPOFFGRADIENTCALCULATOR_H_

#include "rele/algorithms/batch/policy_search/gradient/OffGradientCalculator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class GPOMDPOffGradientCalculator : public OffGradientCalculator<ActionC, StateC>
{
public:
    GPOMDPOffGradientCalculator(RewardTransformation& rewardf, Dataset<ActionC,StateC>& data,
                                Policy<ActionC,StateC>& behaviour, DifferentiablePolicy<ActionC,StateC>& policy,
                                double gamma)
        : OffGradientCalculator<ActionC, StateC>(rewardf, data, behaviour, policy, gamma)

    {

    }

    virtual arma::vec computeGradient(const arma::uvec& indexes = arma::uvec()) override
    {
        int parametersN = this->policy.getParametersSize();

        arma::vec gradient(parametersN, arma::fill::zeros);

        double importanceWeightsAverage = 0;

        int episodeN = this->getEpisodesNumber(indexes);
        for (int i = 0; i < episodeN; ++i)
        {
            //core setup
            unsigned int ep = this->getEpisodeIndex(indexes, i);
            int stepN = this->data[ep].size();
            arma::vec sumGradLog(parametersN, arma::fill::zeros);

            //iterate the episode
            double df = 1.0;
            double targetIW = 1.0;
            double behavoiourIW = 1.0;
            double importanceWeightsSum = 0;

            for (int t = 0; t < stepN; ++t)
            {
                Transition<ActionC, StateC>& tr = this->data[ep][t];

                // compute the reward gradients
                double Rew = df * this->rewardf(tr.r);
                sumGradLog += this->policy.difflog(tr.x, tr.u);
                targetIW *= this->policy(tr.x, tr.u);
                behavoiourIW *= this->behaviour(tr.x, tr.u);
                double iw = targetIW / behavoiourIW;

                gradient += sumGradLog * Rew * iw;

                df *= this->gamma;
            }

            importanceWeightsAverage += importanceWeightsSum / stepN;

        }

        // compute mean values
        gradient /= importanceWeightsAverage;


        return gradient;
    }

    virtual ~GPOMDPOffGradientCalculator()
    {

    }

};

template<class ActionC, class StateC>
class GPOMDPSingleBaseOffGradientCalculator : public OffGradientCalculator<ActionC, StateC>
{
public:
    GPOMDPSingleBaseOffGradientCalculator(RewardTransformation& rewardf, Dataset<ActionC,StateC>& data,
                                          Policy<ActionC,StateC>& behaviour, DifferentiablePolicy<ActionC,StateC>& policy,
                                          double gamma)
        : OffGradientCalculator<ActionC, StateC>(rewardf, data, behaviour, policy, gamma)

    {

    }

    virtual arma::vec computeGradient(const arma::uvec& indexes = arma::uvec()) override
    {
        int parametersN = this->policy.getParametersSize();
        unsigned int maxSteps = this->data.getEpisodeMaxLength();

        int episodeN = this->getEpisodesNumber(indexes);

        // Reset computed results
        arma::vec gradient(parametersN, arma::fill::zeros);

        // gradient basics
        arma::mat Rew_epStep(episodeN, maxSteps);
        arma::mat iw_epStep(episodeN, maxSteps);
        arma::cube sumGradLog_epStep(episodeN, maxSteps, parametersN);
        arma::vec maxsteps_Ep(episodeN);

        // baseline
        arma::vec baseline_num1(parametersN, arma::fill::zeros);
        arma::vec baseline_num2(parametersN, arma::fill::zeros);


        double importanceWeightsAverage = 0;

        for (unsigned int i = 0; i < episodeN; ++i)
        {
            //core setup
            unsigned int ep = this->getEpisodeIndex(indexes, i);
            int stepN = this->data[ep].size();

            arma::vec sumGradLog(parametersN, arma::fill::zeros);

            //iterate the episode
            double df = 1.0;
            double targetIW = 1.0;
            double behavoiourIW = 1.0;
            double importanceWeightsSum = 0;

            for (int t = 0; t < stepN; ++t)
            {
                Transition<ActionC, StateC>& tr = this->data[ep][t];

                // compute the basic elements used to compute the gradients
                double Rew = df * this->rewardf(tr.r);
                sumGradLog += this->policy.difflog(tr.x, tr.u);
                targetIW *= this->policy(tr.x, tr.u);
                behavoiourIW *= this->behaviour(tr.x, tr.u);
                double iw = targetIW / behavoiourIW;
                importanceWeightsSum += iw;

                // store the basic elements used to compute the gradients
                Rew_epStep(i, t) = Rew;
                iw_epStep(i, t) = iw;
                sumGradLog_epStep.tube(i, t) = sumGradLog;

                // compute the baselines
                baseline_num1 += sumGradLog * Rew * iw;
                baseline_num2 += sumGradLog * iw;

                df *= this->gamma;
            }

            // store the actual length of the current episode (<= maxsteps)
            maxsteps_Ep(i) = stepN;
            importanceWeightsAverage += importanceWeightsSum / stepN;

        }

        // compute the gradients
        for (unsigned int ep = 0; ep < episodeN; ++ep)
        {
            // compute the gradients
            arma::vec baseline_num = baseline_num1 % baseline_num2;
            arma::vec baseline_den = baseline_num2 % baseline_num2;
            arma::vec baseline = baseline_num / baseline_den;
            baseline(arma::find_nonfinite(baseline)).zeros();

            for (unsigned int t = 0; t < maxsteps_Ep(ep); ++t)
            {
                arma::vec sumGradLog_ep_t = sumGradLog_epStep.tube(ep,t);
                gradient += sumGradLog_ep_t % (Rew_epStep(ep, t) - baseline) * iw_epStep(ep,t);
            }
        }

        // compute mean values
        gradient /= importanceWeightsAverage;

        return gradient;
    }

    virtual ~GPOMDPSingleBaseOffGradientCalculator()
    {

    }

};



template<class ActionC, class StateC>
class GPOMDPMultyBaseOffGradientCalculator : public OffGradientCalculator<ActionC, StateC>
{
public:
    GPOMDPMultyBaseOffGradientCalculator(RewardTransformation& rewardf, Dataset<ActionC,StateC>& data,
                                         Policy<ActionC,StateC>& behaviour, DifferentiablePolicy<ActionC,StateC>& policy,
                                         double gamma)
        : OffGradientCalculator<ActionC, StateC>(rewardf, data, behaviour, policy, gamma)

    {

    }

    virtual arma::vec computeGradient(const arma::uvec& indexes = arma::uvec()) override
    {
        int parametersN = this->policy.getParametersSize();
        unsigned int maxSteps = this->data.getEpisodeMaxLength();

        int episodeN =  this->getEpisodesNumber(indexes);

        // Reset computed results
        arma::vec gradient(parametersN, arma::fill::zeros);

        // gradient basics
        arma::mat Rew_epStep(episodeN, maxSteps);
        arma::mat iw_epStep(episodeN, maxSteps);
        arma::cube sumGradLog_epStep(episodeN, maxSteps, parametersN);
        arma::vec maxsteps_Ep(episodeN);

        // baseline
        arma::mat baseline_den(parametersN, maxSteps, arma::fill::zeros);
        arma::mat baseline_num(parametersN, maxSteps, arma::fill::zeros);


        double importanceWeightsAverage = 0;

        for (unsigned int i = 0; i < episodeN; ++i)
        {
            //core setup
            unsigned int ep = this->getEpisodeIndex(indexes, i);
            int stepN = this->data[ep].size();

            arma::vec sumGradLog(parametersN, arma::fill::zeros);

            //iterate the episode
            double df = 1.0;
            double targetIW = 1.0;
            double behavoiourIW = 1.0;
            double importanceWeightsSum = 0;

            for (unsigned int t = 0; t < stepN; ++t)
            {
                Transition<ActionC, StateC>& tr = this->data[ep][t];

                // compute the basic elements used to compute the gradients
                double Rew = df * this->rewardf(tr.r);
                sumGradLog += this->policy.difflog(tr.x, tr.u);
                targetIW *= this->policy(tr.x, tr.u);
                behavoiourIW *= this->behaviour(tr.x, tr.u);
                double iw = targetIW / behavoiourIW;
                importanceWeightsSum += iw;

                // store the basic elements used to compute the gradients
                Rew_epStep(i, t) = Rew;
                iw_epStep(i, t) = iw;
                sumGradLog_epStep.tube(i, t) = sumGradLog;

                // compute the baselines
                arma::vec sumGradLog2 = sumGradLog % sumGradLog;
                baseline_num.col(t) += sumGradLog2 * Rew * iw * iw;
                baseline_den.col(t) += sumGradLog2 * iw * iw;

                df *= this->gamma;
            }

            // store the actual length of the current episode (<= maxsteps)
            maxsteps_Ep(i) = stepN;
            importanceWeightsAverage += importanceWeightsSum / stepN;

        }

        // compute the gradients
        for (unsigned int ep = 0; ep < episodeN; ++ep)
        {
            for (unsigned int t = 0; t < maxsteps_Ep(ep); ++t)
            {
                // compute the gradients
                arma::vec baseline_t = baseline_num.col(t) / baseline_den.col(t);
                baseline_t(arma::find_nonfinite(baseline_t)).zeros();

                arma::vec sumGradLog_ep_t = sumGradLog_epStep.tube(ep,t);

                gradient += sumGradLog_ep_t % (Rew_epStep(ep, t) - baseline_t) * iw_epStep(ep,t);
            }
        }

        // compute mean values
        gradient /= importanceWeightsAverage;

        return gradient;
    }

    virtual ~GPOMDPMultyBaseOffGradientCalculator()
    {

    }

};


}


#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_POLICY_SEARCH_GRADIENT_GPOMDPOFFGRADIENTCALCULATOR_H_ */
