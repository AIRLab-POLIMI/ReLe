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

#ifndef INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEARNATURALGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEARNATURALGRADIENTCALCULATOR_H_

#include "rele/IRL/utils/gradient_nonlinear/NonlinearGradientCalculator.h"
#include "rele/IRL/utils/FisherMatrixCalculator.h"

namespace ReLe
{


template<class ActionC, class StateC, class Calculator>
class NonlinearNaturalGradientCalculator : public Calculator
{
    static_assert(std::is_base_of<NonlinearGradientCalculator<ActionC, StateC>, Calculator>::value,
                  "Not valid Calculator class as template parameter");

public:
    NonlinearNaturalGradientCalculator(ParametricRegressor& rewardFunc,
                                       Dataset<ActionC,StateC>& data,
                                       DifferentiablePolicy<ActionC,StateC>& policy,
                                       double gamma):
        Calculator(rewardFunc, data, policy, gamma)
    {
        fisher = FisherMatrixcalculator<ActionC, StateC>::computeFisherMatrix(policy, data);
    }

    virtual ~NonlinearNaturalGradientCalculator()
    {

    }

protected:
    virtual void compute() override
    {
        Calculator::compute();

        unsigned int rnk = arma::rank(fisher);

        if (rnk == fisher.n_rows)
        {
            this->gradient = arma::solve(fisher, this->gradient);
            this->dGradient = arma::solve(fisher, this->dGradient);
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be "
                      << fisher.n_rows << std::endl;

            arma::mat H = arma::pinv(fisher);
            this->gradient = H * this->gradient;
            this->dGradient = H * this->dGradient;
        }
    }

private:
    arma::mat fisher;

};

}


#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEARNATURALGRADIENTCALCULATOR_H_ */
