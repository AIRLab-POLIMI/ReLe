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

#ifndef INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_BLACKBOXOUTPUTDATA_H_
#define INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_BLACKBOXOUTPUTDATA_H_

#include "Basics.h"

#include <vector>
#include <armadillo>

namespace ReLe
{

struct BlackBoxPolicyIndividual
{
    BlackBoxPolicyIndividual(unsigned int nbParams, unsigned int nbEvals) :
        Pparams(nbParams), Jvalues(nbEvals)
    {
    }

    arma::vec Pparams;  //policy parameters
    arma::vec Jvalues;  //policy evaluation (n evaluations for each policy)
};

template<class IndividualsClass>
class BlackBoxOutputData: virtual public AgentOutputData
{

public:
    BlackBoxOutputData(unsigned int nbIndividual, unsigned int nbParams,
                       unsigned int nbEvals) :
        AgentOutputData(true)
    {
        individuals.assign(nbIndividual, IndividualsClass(nbParams, nbEvals));
    }

public:
    arma::vec metaParams;
    std::vector<IndividualsClass> individuals;

};

}


#endif /* INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_BLACKBOXOUTPUTDATA_H_ */
