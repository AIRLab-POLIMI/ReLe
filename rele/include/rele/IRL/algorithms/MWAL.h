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
#include "regressors/LinearApproximator.h"
#include "IRLAlgorithm.h"
#include "IRLSolver.h"

namespace ReLe
{

template<class ActionC, class StateC, class FeaturesInputC = arma::vec>
class MWAL : public IRLAlgorithm<ActionC, StateC>
{
public:
    MWAL(unsigned int T, arma::vec& muE, IRLSolver<ActionC, StateC, FeaturesInputC>& solver)
        : solver(solver), T(T), gamma(solver.getGamma()), muE(muE)
    {
        //Compute beta parameter
        unsigned int k = solver.getBasisSize();
        logBeta = std::log(1.0 / (1 + std::sqrt(2.0*std::log(k)/T)));

        //initialize W(i) = 1 forall i
        W.ones(k);
        std::cout << "W: "<< std::endl << W.t() << std::endl;

        deltaMuNorm = std::numeric_limits<double>::infinity();

        policyOpt = nullptr;
    }

    virtual void run() override
    {
        for(int i = 0; i <= T; i++)
        {
            std::cout << "T = " << i << std::endl;

            //Compute current iteration weights
            std::cout << "Computing weights..."<< std::endl;
            arma::vec w = W / arma::sum(W);
            solver.setWeights(w);

            std::cout << "w: "<< std::endl << w.t() << std::endl;

            //compute optimal policy
            solver.solve();

            //compute feature expectations
            std::cout << "Computing features expectations..."<< std::endl;
            const arma::vec& mu = solver.computeFeaturesExpectations();
            double deltaMuNormNew = arma::norm(mu - muE);

            std::cout << "mu: "<< std::endl << mu.t() << std::endl;
            std::cout << "muE: "<< std::endl << muE.t() << std::endl;
            std::cout << "deltaMuNorm: " << deltaMuNormNew << std::endl;

            if(deltaMuNormNew < deltaMuNorm)
            {
                if(!policyOpt)
                    delete policyOpt;

                deltaMuNorm = deltaMuNormNew;
                policyOpt = solver.getPolicy().clone();
                wOpt = w;
            }

            //update W
            std::cout << "Updating W..."<< std::endl;

            W = W % arma::exp(logBeta * G(mu));

            std::cout << "W: "<< std::endl << W.t() << std::endl;

        }
    }

    virtual arma::vec getWeights() override
    {
        return wOpt;
    }

    virtual Policy<ActionC, StateC>* getPolicy() override
    {
        return policyOpt;
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

    //Solver
    IRLSolver<ActionC, StateC, FeaturesInputC>& solver;

    //Algorithm data
    double logBeta;

    arma::vec W;

    //Best policy data
    double deltaMuNorm;
    arma::vec wOpt;
    Policy<ActionC, StateC>* policyOpt;


};

}

#endif /* INCLUDE_RELE_IRL_ALGORITHMS_MWAL_H_ */
