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

#ifndef INCLUDE_RELE_IRL_UTILS_GRADIENT_EPISODIC_PGPEGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_GRADIENT_EPISODIC_PGPEGRADIENTCALCULATOR_H_

#include "rele/IRL/utils/gradient/episodic/EpisodicGradientCalculator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class PGPEGradientCalculator : public EpisodicGradientCalculator<ActionC, StateC>
{
protected:
    USING_EPISODIC_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    PGPEGradientCalculator(const arma::mat& theta,
                           const arma::mat& phi,
                           DifferentiableDistribution& dist,
                           double gamma)
        : EpisodicGradientCalculator<ActionC, StateC>(theta, phi, dist, gamma)
    {

    }

    virtual ~PGPEGradientCalculator()
    {

    }

protected:
    virtual arma::mat computeGradientDiff() override
    {
        unsigned int N = theta.n_cols;
        arma::mat gradLog = computeGradLog();
        arma::mat gradientDiff = gradLog*phi.t();
        gradientDiff /= N;

        return gradientDiff;
    }

    arma::mat computeGradLog()
    {
        arma::mat gradLog(theta.n_rows, theta.n_cols);

        for(unsigned int i = 0; i < theta.n_cols; i++)
        {
            gradLog.col(i) = dist.difflog(theta.col(i));
        }

        return gradLog;
    }

};

template<class ActionC, class StateC>
class PGPEBaseGradientCalculator : public PGPEGradientCalculator<ActionC, StateC>
{
protected:
    USING_EPISODIC_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    PGPEBaseGradientCalculator(const arma::mat& theta,
                               const arma::mat& phi,
                               DifferentiableDistribution& dist,
                               double gamma)
        : PGPEGradientCalculator<ActionC, StateC>(theta, phi, dist, gamma)
    {

    }

    virtual ~PGPEBaseGradientCalculator()
    {

    }

protected:
    virtual arma::mat computeGradientDiff() override
    {
        //Compute gradient essentials
        unsigned int N = theta.n_cols;
        arma::mat gradLog = this->computeGradLog();
        arma::mat gradLog2 = gradLog % gradLog;

        //compute baseline
        arma::mat baseline_num = (gradLog2)*phi.t();
        arma::vec baseline_den = arma::sum(gradLog2, 1);

        arma::mat baseline = baseline_num.each_col() / baseline_den;
        baseline(arma::find_nonfinite(baseline)).zeros();

        //compute gradient
        arma::mat gradientDiff(theta.n_rows, phi.n_rows, arma::fill::zeros);
        for (int ep = 0; ep < N; ep++)
        {
            for(int r = 0; r < phi.n_rows; r++)
                gradientDiff.col(r) += (phi(r, ep) - baseline.col(r)) % gradLog.col(ep);
        }

        gradientDiff /= N;

        return gradientDiff;
    }


};

}


#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENT_EPISODIC_PGPEGRADIENTCALCULATOR_H_ */
