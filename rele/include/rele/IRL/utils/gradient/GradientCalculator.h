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

#ifndef INCLUDE_RELE_IRL_UTILS_GRADIENTCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_GRADIENTCALCULATOR_H_

#include <armadillo>

namespace ReLe
{

template<class ActionC, class StateC>
class GradientCalculator
{
public:
    GradientCalculator(unsigned int dp, unsigned int dr, double gamma)
        : gamma(gamma), gradientDiff(dp, dr, arma::fill::zeros)
    {
        computed = false;
    }

    arma::vec computeGradient(const arma::vec& w)
    {
        compute();

        return gradientDiff*w;
    }

    arma::mat getGradientDiff()
    {
        compute();

        return gradientDiff;
    }

    virtual ~GradientCalculator()
    {

    }


protected:
    virtual arma::mat computeGradientDiff() = 0;

private:
    void compute()
    {
        if(!computed)
        {
            gradientDiff = computeGradientDiff();
            computed = true;
        }

    }



protected:
    double gamma;

private:
    bool computed;
    arma::mat gradientDiff;

};


}


#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENTCALCULATOR_H_ */
