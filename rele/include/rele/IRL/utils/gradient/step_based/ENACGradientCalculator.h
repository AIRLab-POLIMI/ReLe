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

#ifndef INCLUDE_RELE_IRL_UTILS_GRADIENT_ENACGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_GRADIENT_ENACGRADIENTCALCULATOR_H_

#include "rele/IRL/utils/gradient/step_based/StepBasedGradientCalculator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class ENACGradientCalculator: public StepBasedGradientCalculator<ActionC, StateC>
{
protected:
    USING_STEP_BASED_CALCULATORS_MEMBERS(ActionC, StateC)
public:
    ENACGradientCalculator(Features& phi, Dataset<ActionC, StateC>& data,
                           DifferentiablePolicy<ActionC, StateC>& policy, double gamma) :
        StepBasedGradientCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~ENACGradientCalculator()
    {

    }

protected:
    virtual arma::mat computeGradientDiff() override
    {
        int dp = policy.getParametersSize();
        int dr = phi.rows();
        arma::mat Rew = data.computeEpisodeFeatureExpectation(phi, gamma);

        arma::mat g(dp + 1, dr, arma::fill::zeros);
        arma::mat fisher(dp + 1, dp + 1, arma::fill::zeros);

        // compute gradient and extended fisher matrix
        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            arma::vec psi(dp + 1);

            psi.rows(0, dp - 1) = this->computeSumGradLog(data[i]);
            psi(dp) = 1.0;

            fisher += psi * psi.t();
            g += psi * Rew.col(i).t();
        }

        // compute mean value
        fisher /= nbEpisodes;
        g /= nbEpisodes;

        arma::mat nat_grad;
        int rnk = arma::rank(fisher);
        if (rnk == fisher.n_rows)
        {
            nat_grad = arma::solve(fisher, g);
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk
                      << ")!!! Should be " << fisher.n_rows << std::endl;

            arma::mat H = arma::pinv(fisher);
            nat_grad = H * g;
        }

        return nat_grad.rows(0, dp - 1);
    }
};

template<class ActionC, class StateC>
class ENACBaseGradientCalculator: public StepBasedGradientCalculator<ActionC, StateC>
{
protected:
    USING_STEP_BASED_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    ENACBaseGradientCalculator(Features& phi, Dataset<ActionC, StateC>& data,
                               DifferentiablePolicy<ActionC, StateC>& policy, double gamma) :
        StepBasedGradientCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~ENACBaseGradientCalculator()
    {

    }

protected:
    virtual arma::mat computeGradientDiff() override
    {
        int dp = policy.getParametersSize();
        int dr = phi.rows();
        arma::mat Rew = data.computeEpisodeFeatureExpectation(phi, gamma);

        arma::mat g(dp + 1, dr, arma::fill::zeros);
        arma::mat fisher(dp + 1, dp + 1, arma::fill::zeros);
        arma::vec Jpol(dr, arma::fill::zeros);
        arma::vec eligibility(dp + 1, arma::fill::zeros);

        // compute gradient and extended fisher matrix
        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {

            arma::vec psi(dp + 1);
            psi.rows(0, dp - 1) = this->computeSumGradLog(data[i]);
            psi(dp) = 1.0;

            fisher += psi * psi.t();
            g += psi * Rew.col(i).t();
            eligibility += psi;
            Jpol += Rew.col(i);
        }

        // compute mean value
        fisher /= nbEpisodes;
        g /= nbEpisodes;
        eligibility /= nbEpisodes;
        Jpol /= nbEpisodes;

        arma::mat nat_grad;
        int rnk = arma::rank(fisher);

        if (rnk == fisher.n_rows)
        {
            arma::mat tmp = arma::solve(
                                nbEpisodes * fisher - eligibility * eligibility.t(),
                                eligibility);
            double Q = (1 + arma::as_scalar(eligibility.t() * tmp)) / nbEpisodes;
            arma::rowvec b = Q * (Jpol.t() - eligibility.t() * arma::solve(fisher, g));
            arma::mat grad = g - eligibility * b;
            nat_grad = arma::solve(fisher, grad);
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk
                      << ")!!! Should be " << fisher.n_rows << std::endl;

            arma::mat H = arma::pinv(fisher);
            arma::mat tmp = arma::pinv(nbEpisodes * fisher - eligibility * eligibility.t());
            double Q = 1 + arma::as_scalar(eligibility.t() * tmp * eligibility);
            arma::rowvec b = Q * (Jpol.t() - eligibility.t() * H * g) / nbEpisodes;
            arma::mat grad = g - eligibility * b;
            nat_grad = H * (grad);
        }

        return nat_grad.rows(0, dp - 1);
    }
};

}

#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENT_ENACGRADIENTCALCULATOR_H_ */
