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

#ifndef INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEAR_NONLINEARENACCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEAR_NONLINEARENACCALCULATOR_H_

#include "rele/IRL/utils/gradient_nonlinear/NonlinearGradientCalculator.h"


namespace ReLe
{

template<class ActionC, class StateC>
class NonlinearENACCalculator : public NonlinearGradientCalculator<ActionC,StateC>
{
protected:
    USE_NONLINEAR_GRADIENT_CALCULATOR_MEMBERS(ActionC, StateC)

public:
    NonlinearENACCalculator(ParametricRegressor& rewardFunc,
                            Dataset<ActionC,StateC>& data,
                            DifferentiablePolicy<ActionC,StateC>& policy,
                            double gamma) : NonlinearGradientCalculator<ActionC,StateC>(rewardFunc, data, policy, gamma)
    {

    }

    virtual void compute() override
    {
        int dp = policy.getParametersSize();
        int dr = rewardFunc.getParametersSize();

        double Rew;
        arma::vec g(dp + 1, arma::fill::zeros), psi(dp + 1);
        arma::mat dg(dp+1, dr, arma::fill::zeros);
        arma::mat fisher(dp + 1, dp + 1, arma::fill::zeros);

        unsigned int nbEpisodes = data.size();
        unsigned int totSteps;


        for (int i = 0; i < nbEpisodes; ++i)
        {
            double Rew;
            arma::rowvec dRew(dr);
            arma::vec sumGradLog(dp);
            this->computeEpisodeStatistics(data[i], Rew, dRew, sumGradLog);

            psi.rows(0, dp-1) = sumGradLog;
            psi(dp) = 1.0;

            fisher += psi * psi.t();
            g += psi * Rew;
            dg += psi * dRew;

            totSteps += data[i].size();
        }

        arma::vec nat_grad;
        arma::mat nat_grad_diff;
        int rnk = arma::rank(fisher);
        if (rnk == fisher.n_rows)
        {
            nat_grad = arma::solve(fisher, g);
            nat_grad_diff = arma::solve(fisher, dg);
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk
                      << ")!!! Should be " << fisher.n_rows << std::endl;

            arma::mat H = arma::pinv(fisher);
            nat_grad = H * g;
            nat_grad_diff = H * dg;
        }

        gradient = nat_grad.rows(0, dp - 1);
        dGradient = nat_grad_diff.rows(0, dp -1);
    }

    virtual ~NonlinearENACCalculator()
    {

    }
};

template<class ActionC, class StateC>
class NonlinearENACBaseCalculator : public NonlinearGradientCalculator<ActionC,StateC>
{
public:
    NonlinearENACBaseCalculator(ParametricRegressor& rewardFunc,
                                Dataset<ActionC,StateC>& data,
                                DifferentiablePolicy<ActionC,StateC>& policy,
                                double gamma) : NonlinearGradientCalculator<ActionC,StateC>(rewardFunc, data, policy, gamma)
    {

    }

    virtual void compute() override
    {
        //TODO implement
    }

    virtual ~NonlinearENACBaseCalculator()
    {

    }
};



}


#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEAR_NONLINEARENACCALCULATOR_H_ */
