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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_SDPEGIRL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_SDPEGIRL_H_

#include "rele/IRL/utils/EpisodicGradientCalculatorFactory.h"
#include "EpisodicLinearIRLAlgorithm.h"

namespace ReLe
{

template<class ActionC, class StateC>
class SDPEGIRL: public EGIRL<ActionC, StateC>
{
public:
    SDPEGIRL(Dataset<ActionC, StateC>& data, const arma::mat& theta, DifferentiableDistribution& dist,
             LinearApproximator& rewardf, double gamma, IrlEpGrad gtype, IrlEpHess htype, double eps = 0.1)
        : EGIRL<ActionC, StateC>(data, theta, dist, rewardf, gamma, gtype), htype(htype), eps(eps)
    {
        hessianCalculator = EpisodicHessianCalculatorFactory<ActionC, StateC>::build(htype, theta, this->phi, dist, gamma);

        std::cout << "positive definite features" << std::endl;
        arma::cube diff = hessianCalculator->getHessianDiff();
        for(unsigned int i = 0; i < diff.n_slices; i++)
        {
        	double maxEig = arma::max(arma::eig_sym(diff.slice(i)));

        	if(maxEig < 0)
        		std::cout << maxEig << " " << i << std::endl;
        }
    }

    virtual ~SDPEGIRL()
    {

    }

protected:
    virtual void setupOptimization(unsigned int effective_dim, unsigned int maxFunEvals) override
    {
    	this->optAlg = nlopt::LD_SLSQP;
        LinearIRLAlgorithm<ActionC, StateC>::setupOptimization(effective_dim, maxFunEvals);

        this->optimizator.add_inequality_constraint(sdConstraint, this, 0.0);
    }

    static double sdConstraint(unsigned int n, const double *x,
                               double *grad, void *data)
    {
        auto& self = *static_cast<SDPEGIRL<ActionC,StateC>*>(data);

        arma::vec parV(const_cast<double*>(x), n, true);
        arma::vec&& w = self.simplex.reconstruct(parV);

        arma::mat hessian = self.hessianCalculator->computeHessian(w);

        arma::vec lambda;
        arma::mat V;
        arma::eig_sym(lambda, V, hessian);

        double lambdaMax = arma::as_scalar(lambda.tail(1));
        arma::vec vi = V.tail_cols(1);

        if(grad)
        {
            arma::cube diff = self.hessianCalculator->getHessianDiff();
            unsigned int indx_n = self.simplex.getFeatureIndex(n);
            arma::mat diff_n = diff.slice(indx_n);
            for(unsigned int i = 0; i < n; i++)
            {
                unsigned int indx = self.simplex.getFeatureIndex(i);
                arma::mat diff_i = diff.slice(indx);
                grad[i] = arma::as_scalar(vi.t()*(diff_i-diff_n)*vi);
            }
        }



        return lambdaMax+self.eps;
    }

protected:
    IrlEpHess htype;
    double eps;
    HessianCalculator<ActionC, StateC>* hessianCalculator;
};



}




#endif /* INCLUDE_RELE_IRL_ALGORITHMS_SDPEGIRL_H_ */
