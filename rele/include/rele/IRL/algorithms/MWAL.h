/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_MWAL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_MWAL_H_

#include "BasisFunctions.h"
#include "LinearApproximator.h"
#include "IRLAlgorithm.h"

namespace ReLe
{

template<class ActionC, class StateC>
class MWAL : public IRLAlgorithm<ActionC, StateC>
{
public:
    MWAL(AbstractBasisVector& basis, LinearApproximator& regressor, Core<ActionC, StateC>& core,
         unsigned int T, double gamma, arma::vec& muE)
        : basis(basis), regressor(regressor), core(core), T(T), gamma(gamma), muE(muE)
    {
        //Compute beta parameter
        unsigned int k = basis.size();
        logBeta = std::log(1.0 / (1 + std::sqrt(2.0*std::log(k)/T)));

        //initialize W(i) = 1 forall i
        W.ones(k);
        std::cout << "W: "<< std::endl << W.t() << std::endl;
    }

    virtual void run()
    {
        for(int i = 0; i <= T; i++)
        {
            std::cout << "T = " << i << std::endl;

            //Compute current iteration weights
            std::cout << "Computing weights..."<< std::endl;
            arma::vec w = W / arma::sum(W);
            regressor.setParameters(w);
            std::cout << "w: "<< std::endl << w.t() << std::endl;

            //compute optimal policy
            std::cout << "Computing optimal policy..."<< std::endl;
            EmptyStrategy<ActionC, StateC> emptyStrategy;
            core.getSettings().loggerStrategy = &emptyStrategy;
            core.runEpisodes();

            //compute feature expectations
            std::cout << "Computing features expectations..."<< std::endl;
            CollectorStrategy<ActionC, StateC> strategy;
            core.getSettings().loggerStrategy = &strategy;
            core.runTestEpisodes();

            const arma::vec& mu = strategy.data.computefeatureExpectation(basis, gamma);
            std::cout << "mu: "<< std::endl << mu.t() << std::endl;
            std::cout << "muE: "<< std::endl << muE.t() << std::endl;
            std::cout << "deltaMuNorm: " << arma::norm(mu - muE) << std::endl;

            //update W
            std::cout << "Updating W..."<< std::endl;
            W = W % arma::exp(logBeta * G(mu));
            std::cout << "W: "<< std::endl << W.t() << std::endl;

        }
    }

    virtual arma::vec getWeights()
    {
        return W / arma::sum(W);
    }

    virtual Policy<ActionC, StateC>* getPolicy()
    {
        return nullptr;
    }

    virtual ~MWAL()
    {

    }

private:
    inline arma::vec G(const arma::vec& mu)
    {
        return ((1.0 - gamma)*(mu - muE) + 2.0) / 4.0;
    }


private:
    //Algorithms parameters
    const unsigned int T;
    const double gamma;
    const arma::vec& muE;

    //Data needed to compute policy and feature expectation
    AbstractBasisVector& basis;
    LinearApproximator& regressor;
    Core<ActionC, StateC>& core;

    //Algorithm data
    double logBeta;

    arma::vec W;

};

}

#endif /* INCLUDE_RELE_IRL_ALGORITHMS_MWAL_H_ */
