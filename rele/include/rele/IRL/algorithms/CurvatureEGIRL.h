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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_CURVATUREEGIRL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_CURVATUREEGIRL_H_

#include "rele/IRL/utils/EpisodicGradientCalculatorFactory.h"
#include "rele/IRL/utils/EpisodicHessianCalculatorFactory.h"
#include "EpisodicLinearIRLAlgorithm.h"

namespace ReLe
{

template<class ActionC, class StateC>
class CurvatureEGIRL: public EGIRL<ActionC, StateC>
{
public:
    CurvatureEGIRL(Dataset<ActionC, StateC>& data, const arma::mat& theta, DifferentiableDistribution& dist,
                   LinearApproximator& rewardf, double gamma, IrlEpGrad gtype, IrlEpHess htype, double eps = 0.025)
        : EGIRL<ActionC, StateC>(data, theta, dist, rewardf, gamma, gtype), htype(htype), eps(eps)
    {
        hessianCalculator = EpisodicHessianCalculatorFactory<ActionC, StateC>::build(htype, theta, this->phi, dist, gamma);
    }

    virtual ~CurvatureEGIRL()
    {

    }

    virtual void run() override
    {
        EGIRL<ActionC, StateC>::run();

        arma::vec w = this->rewardf.getParameters();

        arma::mat hessian = this->hessianCalculator->computeHessian(w);
        arma::vec gradient = this->gradientCalculator->computeGradient(w);

        double curvature = arma::as_scalar(gradient.t()*hessian*gradient);

        std::cout << "curvature: " <<  curvature << std::endl;
        std::cout << "eps: " << eps << std::endl;

    }

protected:
    virtual void setupOptimization(unsigned int effective_dim, unsigned int maxFunEvals) override
    {
        this->optAlg = nlopt::LD_SLSQP;
        LinearIRLAlgorithm<ActionC, StateC>::setupOptimization(effective_dim, maxFunEvals);

        this->optimizator.add_inequality_constraint(curvatureConstraint, this, 0.0);
    }

    static double curvatureConstraint(unsigned int n, const double *x,
                                      double *grad, void *data)
    {
        auto& self = *static_cast<CurvatureEGIRL<ActionC,StateC>*>(data);

        arma::vec parV(const_cast<double*>(x), n, true);
        arma::vec&& w = self.simplex.reconstruct(parV);

        arma::mat hessian = self.hessianCalculator->computeHessian(w);
        arma::vec gradient = self.gradientCalculator->computeGradient(w);

        double curvature = arma::as_scalar(gradient.t()*hessian*gradient);

        if(grad)
        {
            arma::cube diffH = self.hessianCalculator->getHessianDiff();
            arma::mat diffG = self.gradientCalculator->getGradientDiff();

            unsigned int indx_n = self.simplex.getFeatureIndex(n);
            arma::mat diffH_n = diffH.slice(indx_n);
            arma::vec diffG_n = diffG.col(indx_n);
            for(unsigned int i = 0; i < n; i++)
            {
                unsigned int indx = self.simplex.getFeatureIndex(i);
                arma::mat diffH_i = diffH.slice(indx)-diffH_n;
                arma::vec diffG_i = diffG.col(indx)-diffG_n;
                grad[i] = arma::as_scalar(diffG_i.t()*hessian*gradient)
                          + arma::as_scalar(gradient.t()*diffH_i*gradient)
                          + arma::as_scalar(gradient.t()*hessian*diffG_i);
            }
        }

        return curvature+self.eps;
    }

protected:
    IrlEpHess htype;
    double eps;
    HessianCalculator<ActionC, StateC>* hessianCalculator;
};



}




#endif /* INCLUDE_RELE_IRL_ALGORITHMS_CURVATUREEGIRL_H_ */
