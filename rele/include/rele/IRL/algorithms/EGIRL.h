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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_EGIRL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_EGIRL_H_

#include "EpisodicLinearIRLAlgorithm.h"

namespace ReLe
{

template<class ActionC, class StateC>
class EGIRL: public EpisodicLinearIRLAlgorithm<ActionC, StateC>
{
public:
    EGIRL(Dataset<ActionC, StateC>& data, const arma::mat& theta,
          DifferentiableDistribution& dist, LinearApproximator& rewardf, double gamma)
        : EpisodicLinearIRLAlgorithm<Actionc, StateC>(data, theta, rewardf, gamma)
    {

    }

    virtual double objFunction(const arma::vec& xSimplex, arma::vec& df)
    {

    }


    virtual ~EGIRL()
    {

    }
};



}


#endif /* INCLUDE_RELE_IRL_ALGORITHMS_EGIRL_H_ */
