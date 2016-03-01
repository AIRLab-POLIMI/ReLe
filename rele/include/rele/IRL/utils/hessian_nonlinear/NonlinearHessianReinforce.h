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

#ifndef INCLUDE_RELE_IRL_UTILS_HESSIAN_NONLINEAR_NONLINEARHESSIANREINFORCE_H_
#define INCLUDE_RELE_IRL_UTILS_HESSIAN_NONLINEAR_NONLINEARHESSIANREINFORCE_H_

#include "rele/IRL/utils/hessian_nonlinear/NonlinearHessianCalculator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class NonlinearHessianReinforce : public NonlinearHessianCalculator<ActionC, StateC>
{
public:
    NonlinearHessianReinforce(ParametricRegressor& rewardFunc,
                              Dataset<ActionC, StateC>& data,
                              DifferentiablePolicy<ActionC, StateC>& policy, double gamma) :
        NonlinearHessianCalculator<ActionC, StateC>(rewardFunc, data, policy, gamma)
    {

    }

    virtual void compute() override
    {
        unsigned int dp = this->policy.getParametersSize();
        unsigned int dr = this->rewardFunc.getParametersSize();
        this->H.zeros(dp, dp);
        this->Hdiff.zeros(dp, dp, dr);

        for (auto& episode : this->data)
        {

            double Rew = 0;
            arma::vec dRew(dr, arma::fill::zeros);
            arma::vec sumGradLog(dp, arma::fill::zeros);
            arma::mat sumHessLog(dp, dp, arma::fill::zeros);

            this->computeEpisodeStatistics(episode, Rew, dRew, sumGradLog, sumHessLog);

            arma::mat G = sumGradLog*sumGradLog.t()+sumHessLog;

            this->H += Rew*G;

            for(unsigned int r = 0; r < dr; r++)
                this->Hdiff.slice(r) += dRew(r) * G;
        }

        this->H /= this->data.size();
        this->Hdiff /= this->data.size();

    }

    virtual ~NonlinearHessianReinforce()
    {

    }
};


template<class ActionC, class StateC>
class NonlinearHessianReinforceBase : public NonlinearHessianCalculator<ActionC, StateC>
{
public:
    NonlinearHessianReinforceBase(ParametricRegressor& rewardFunc,
                                  Dataset<ActionC, StateC>& data,
                                  DifferentiablePolicy<ActionC, StateC>& policy, double gamma) :
        NonlinearHessianCalculator<ActionC, StateC>(rewardFunc, data, policy, gamma)
    {

    }

    virtual void compute() override
    {
        unsigned int dp = this->policy.getParametersSize();
        unsigned int dr = this->rewardFunc.getParametersSize();
        unsigned int episodeN = this->data.size();

        this->H.zeros(dp, dp);
        this->Hdiff.zeros(dp, dp, dr);

        arma::vec Rew_ep(episodeN, arma::fill::zeros);
        arma::mat dRew_ep(dr, episodeN, arma::fill::zeros);
        arma::mat baseline_num_Rew(dp, dp, arma::fill::zeros);
        arma::cube baseline_num_dRew(dp, dp, dr, arma::fill::zeros);
        arma::mat baseline_den(dp, dp, arma::fill::zeros);
        arma::cube G_ep(dp, dp, episodeN);

        for(unsigned int ep = 0; ep < episodeN; ep++)
        {
            double Rew;
            arma::vec dRew(dr, arma::fill::zeros);
            arma::vec sumGradLog(dp, arma::fill::zeros);
            arma::mat sumHessLog(dp, dp, arma::fill::zeros);

            this->computeEpisodeStatistics(this->data[ep], Rew, dRew, sumGradLog, sumHessLog);

            // compute hessian essential
            arma::mat G = sumGradLog*sumGradLog.t()+sumHessLog;
            arma::mat G2 = G % G;

            // store hessian essentials
            Rew_ep(ep) = Rew;
            dRew_ep.col(ep) = dRew;
            G_ep.slice(ep) = G;

            // Compute baselines
            baseline_den += G2;
            baseline_num_Rew += Rew*G2;

            for(unsigned int r = 0; r < dr; r++)
                baseline_num_dRew.slice(r) += dRew(r)*G2;
        }

        // compute the hessian
        arma::mat baseline_Rew = baseline_num_Rew / baseline_den;
        baseline_Rew(arma::find_nonfinite(baseline_Rew)).zeros();

        arma::cube baseline_dRew = baseline_num_dRew.each_slice() / baseline_den;
        baseline_dRew(arma::find_nonfinite(baseline_dRew)).zeros();

        for (int ep = 0; ep < episodeN; ep++)
        {
            this->H += (Rew_ep(ep) - baseline_Rew) % G_ep.slice(ep);

            for(int r = 0; r < dr; r++)
                this->Hdiff.slice(r) += (dRew_ep(r, ep) - baseline_dRew.slice(r)) % G_ep.slice(ep);
        }

        // compute mean values
        this->Hdiff /= episodeN;
        this->H /= episodeN;

    }

    virtual ~NonlinearHessianReinforceBase()
    {

    }
};

}

#endif /* INCLUDE_RELE_IRL_UTILS_HESSIAN_NONLINEAR_NONLINEARHESSIANREINFORCE_H_ */
