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

#ifndef INCLUDE_RELE_IRL_UTILS_HESSIAN_EPISODIC_HESSIANPGPE_H_
#define INCLUDE_RELE_IRL_UTILS_HESSIAN_EPISODIC_HESSIANPGPE_H_

#include "rele/IRL/utils/hessian/episodic/EpisodicHessianCalculator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class PGPEHessianCalculator : public EpisodicHessianCalculator<ActionC, StateC>
{
protected:
    USING_EPISODIC_H_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    PGPEHessianCalculator(const arma::mat& theta,
                          const arma::mat& phi,
                          DifferentiableDistribution& dist,
                          double gamma)
        : EpisodicHessianCalculator<ActionC, StateC>(theta, phi, dist, gamma)
    {

    }

    virtual ~PGPEHessianCalculator()
    {

    }

protected:
    virtual arma::cube computeHessianDiff() override
    {
        unsigned int N = theta.n_cols;
        arma::cube G = computeG();
        arma::cube hessianDiff(theta.n_rows, theta.n_rows, phi.n_rows, arma::fill::zeros);
        for (int ep = 0; ep < N; ep++)
        {
            for(int r = 0; r < phi.n_rows; r++)
                hessianDiff.slice(r) += phi(r, ep) * G.slice(ep);
        }


        hessianDiff /= N;

        return hessianDiff;
    }

    arma::cube computeG()
    {
        unsigned int N = theta.n_cols;
        arma::mat gradLog;
        arma::cube grad2Log;
        computeGradLog(gradLog, grad2Log);
        arma::cube G(theta.n_rows, theta.n_rows, theta.n_cols);

        for(unsigned int ep = 0; ep < N; ep++)
        {
            auto gLog = gradLog.col(ep);
            G.slice(ep) += gLog*gLog.t()+grad2Log.slice(ep);
        }

        return G;
    }

    void computeGradLog(arma::mat& gradLog, arma::cube& grad2Log)
    {
        gradLog.set_size(theta.n_rows, theta.n_cols);
        grad2Log.set_size(theta.n_rows, theta.n_rows, theta.n_cols);

        for(unsigned int i = 0; i < theta.n_cols; i++)
        {
            gradLog.col(i) = dist.difflog(theta.col(i));
            grad2Log.slice(i) = dist.diff2log(theta.col(i));
        }
    }

};

template<class ActionC, class StateC>
class PGPEBaseHessianCalculator : public PGPEHessianCalculator<ActionC, StateC>
{
protected:
    USING_EPISODIC_H_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    PGPEBaseHessianCalculator(const arma::mat& theta,
                              const arma::mat& phi,
                              DifferentiableDistribution& dist,
                              double gamma)
        : PGPEHessianCalculator<ActionC, StateC>(theta, phi, dist, gamma)
    {

    }

    virtual ~PGPEBaseHessianCalculator()
    {

    }

protected:
    virtual arma::cube computeHessianDiff() override
    {
        //Compute hessian essentials
        unsigned int N = theta.n_cols;
        arma::cube G = this->computeG();
        arma::cube G2 = G % G;

        //compute baseline
        arma::cube baseline_num(theta.n_rows, theta.n_rows, phi.n_rows, arma::fill::zeros);
        arma::mat baseline_den(theta.n_rows, theta.n_rows);

        for (int ep = 0; ep < N; ep++)
        {
            baseline_den += G2.slice(ep);

            for(int r = 0; r < phi.n_rows; r++)
                baseline_num.slice(r) += phi(r, ep) * G2.slice(ep);
        }

        arma::cube baseline = baseline_num.each_slice() / baseline_den;
        baseline(arma::find_nonfinite(baseline)).zeros();

        //compute hessian
        arma::cube hessianDiff(theta.n_rows, theta.n_rows, phi.n_rows, arma::fill::zeros);
        for (int ep = 0; ep < N; ep++)
        {
            for(int r = 0; r < phi.n_rows; r++)
                hessianDiff.slice(r) += (phi(r, ep) - baseline.slice(r)) % G.slice(ep);
        }

        hessianDiff /= N;

        return hessianDiff;
    }


};

}


#endif /* INCLUDE_RELE_IRL_UTILS_HESSIAN_EPISODIC_HESSIANPGPE_H_ */
