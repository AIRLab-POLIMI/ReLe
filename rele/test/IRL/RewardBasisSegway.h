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

#ifndef TEST_IRL_REWARDBASISSEGWAY_H_
#define TEST_IRL_REWARDBASISSEGWAY_H_

#include "rele/approximators/basis/QuadraticBasis.h"

namespace ReLe
{

class Segway_RewardBasis : public BasisFunction
{
public:
    Segway_RewardBasis(unsigned int i, unsigned int dim)
    {
        arma::mat Q(dim, dim, arma::fill::zeros);
        Q(i, i) = dim;
        bf = new QuadraticBasis(Q, arma::span(0, dim-1));
    }

    virtual double operator()(const arma::vec& input) override
    {
        return -(*bf)(input);
    }

    virtual void writeOnStream(std::ostream& out) override
    {

    }

    virtual void readFromStream(std::istream& in) override
    {

    }

private:
    BasisFunction* bf;

};

}

#endif /* TEST_IRL_REWARDBASISSEGWAY_H_ */
