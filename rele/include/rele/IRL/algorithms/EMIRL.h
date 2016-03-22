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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_EMIRL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_EMIRL_H_

#include "rele/IRL/algorithms/EpisodicLinearIRLAlgorithm.h"

namespace ReLe
{

template<class ActionC, class StateC>
class EMIRL: public EpisodicLinearIRLAlgorithm<ActionC, StateC>
{
    using EpisodicLinearIRLAlgorithm<ActionC, StateC>::theta;
    using EpisodicLinearIRLAlgorithm<ActionC, StateC>::phiBar;

public:
    EMIRL(Dataset<ActionC, StateC>& data, const arma::mat& theta, ParametricNormal& dist,
          LinearApproximator& rewardFunction, double gamma)
        : EpisodicLinearIRLAlgorithm<ActionC, StateC>(data, theta, rewardFunction, gamma),
          wBar(dist.getMean()), sigmaInv(arma::inv(dist.getCovariance()))
    {

    }

    virtual double objFunction(const arma::vec& xSimplex, arma::vec& df) override
    {
        //Compute expectation-maximization update
        arma::vec&& omega = this->simplex.reconstruct(xSimplex);
        arma::vec Jep = this->phiBar.t()*omega;
        double maxJep = arma::max(Jep);
        Jep = Jep - maxJep; //Numerical trick
        arma::vec a = arma::exp(Jep);
        a /= arma::sum(a);

        arma::vec what = theta*a;

        //Compute Kullback Leiber divergence
        arma::vec delta = what - wBar;
        double KL = arma::as_scalar(delta.t()*sigmaInv*delta);

        //Compute derivative
        arma::mat dwhat = theta*(arma::diagmat(a) - a*a.t())*phiBar.t();
        arma::vec dKL = 2*dwhat.t()*sigmaInv*delta;

        df = this->simplex.diff(dKL);

        /*std::cout << "-------------------------------------" << std::endl;
        std::cout << "Jep Max" << std::endl << maxJep << std::endl;
        std::cout << "Jep Min" << std::endl << std::endl << arma::min(Jep) << std::endl;
        std::cout << "a Min" << std::endl << a.min() << std::endl;
        std::cout << "a Max" << std::endl << a.max() << std::endl;
        std::cout << "a mean" << std::endl << arma::mean(a) << std::endl;
        std::cout << "dwhat Min" << std::endl << dwhat.min() << std::endl;
        std::cout << "dwhat Max" << std::endl << dwhat.max() << std::endl;
        std::cout << "rank(dwhat) " << std::endl << arma::rank(dwhat) << std::endl;
        std::cout << "delta" << std::endl << delta.t() << std::endl;
        std::cout << "what" << std::endl << what.t() << std::endl;
        std::cout << "dKL" << std::endl << dKL.t() << std::endl;
        std::cout << "df" << std::endl << df.t() << std::endl;*/

        return KL;

    }

    virtual ~EMIRL()
    {

    }

private:
    arma::vec wBar;
    arma::mat sigmaInv;
};

}

#endif /* INCLUDE_RELE_IRL_ALGORITHMS_EMIRL_H_ */
