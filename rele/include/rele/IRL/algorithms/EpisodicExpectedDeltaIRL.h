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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_EPISODICEXPECTEDDELTAIRL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_EPISODICEXPECTEDDELTAIRL_H_

#include "rele/IRL/utils/EpisodicHessianCalculatorFactory.h"
#include "rele/IRL/utils/EpisodicGradientCalculatorFactory.h"
#include "rele/IRL/algorithms/EpisodicLinearIRLAlgorithm.h"

namespace ReLe
{

template<class ActionC, class StateC>
class EpisodicExpectedDeltaIRL: public EpisodicLinearIRLAlgorithm<ActionC, StateC>
{
public:
    EpisodicExpectedDeltaIRL(Dataset<ActionC, StateC>& data, const arma::mat& theta, DifferentiableDistribution& dist,
                             LinearApproximator& rewardf, double gamma, IrlEpGrad type, IrlEpHess htype) :
        EpisodicLinearIRLAlgorithm<ActionC, StateC>(data, theta, rewardf, gamma)
    {
        Features& features = rewardf.getFeatures();
        phi = data.computeEpisodeFeatureExpectation(features, gamma);

        gradientCalculator = EpisodicGradientCalculatorFactory<ActionC, StateC>::build(type, theta, phi, dist, gamma);
        hessianCalculator = EpisodicHessianCalculatorFactory<ActionC, StateC>::build(htype, theta, phi, dist, gamma);

        this->optAlg = nlopt::algorithm::LN_COBYLA;
    }

    virtual ~EpisodicExpectedDeltaIRL()
    {

    }

    //======================================================================
    // OBJECTIVE FUNCTION
    //----------------------------------------------------------------------

    virtual double objFunction(const arma::vec& xSimplex, arma::vec& df) override
    {

        ++this->nbFunEvals;

        // compute parameters vector
        arma::vec x = this->simplex.reconstruct(xSimplex);

        // compute gradient, hessian and covariance
        arma::vec g = gradientCalculator->computeGradient(x);
        arma::mat H = hessianCalculator->computeHessian(x);
        arma::mat Sigma(H.n_rows, H.n_cols, arma::fill::eye);

        // compute signed hessian
        arma::mat V;
        arma::vec Lambda;
        arma::eig_sym(Lambda, V, H);

        arma::mat Hs = V*arma::diagmat(arma::abs(Lambda))*V.i();

        // Compute function
        double f_linear = arma::as_scalar(g.t() * arma::inv(Hs) * g);
        double f_quadratic = 0.5 * arma::as_scalar(g.t() * arma::inv(H) * g);
        double f_trace = 0.5*arma::trace(H * Sigma)*1e-3;
        double f = f_linear + f_quadratic + f_trace;

        arma::vec e = arma::eig_sym(H);

        // print stuff
        /*std::cout << "f: " << f << std::endl;
        std::cout << "g: " << g.t() << std::endl;
        std::cout << "e: " << e.t() << std::endl;
        std::cout << "f_linear: " << f_linear << std::endl;
        std::cout << "f_quadratic: " << f_quadratic << std::endl;
        std::cout << "f_trace: " << f_trace << std::endl;
        std::cout << "-----------------------------------------" << std::endl;*/

        return f;

    }

private:
    arma::mat phi;

    GradientCalculator<ActionC, StateC>* gradientCalculator;
    HessianCalculator<ActionC, StateC>* hessianCalculator;
};

}



#endif /* INCLUDE_RELE_IRL_ALGORITHMS_EPISODICEXPECTEDDELTAIRL_H_ */
