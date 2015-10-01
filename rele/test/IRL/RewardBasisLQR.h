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

#ifndef SRC_TEST_IRL_GIRL_REWARDBASISLQR_H_
#define SRC_TEST_IRL_GIRL_REWARDBASISLQR_H_

#include "basis/QuadraticBasis.h"

namespace ReLe
{

class LQR_RewardBasis : public BasisFunction
{
public:
    LQR_RewardBasis(unsigned int i, unsigned int dim)
    {
        arma::mat Q(dim, dim);
        arma::mat R(dim, dim);

        double e = 0.1;
        for (int j = 0; j < dim; j++)
        {
            if (i == j)
            {
                Q(j,j) = 1.0 - e;
                R(j,j) = e;
            }
            else
            {
                Q(j,j) = e;
                R(j,j) = 1.0 - e;
            }
        }

        bf1 = new QuadraticBasis(Q, arma::span(0, dim-1));
        bf2 = new QuadraticBasis(R, arma::span(dim,2*dim-1));
    }

    virtual double operator()(const arma::vec& input) override
    {
        return -(*bf1)(input)-(*bf2)(input);
    }

    virtual void writeOnStream(std::ostream& out) override
    {

    }

    virtual void readFromStream(std::istream& in) override
    {

    }

private:
    BasisFunction* bf1;
    BasisFunction* bf2;
};

}


#endif /* SRC_TEST_IRL_GIRL_REWARDBASISLQR_H_ */
