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

#include <cassert>

namespace ReLe
{

template<class ActionC, class StateC>
class NonlinearGradientCalculator
{
public:
    NonlinearGradientCalculator(Regressor& rewardFunc,
                                Dataset<ActionC,StateC>& data,
                                DifferentiablePolicy<ActionC,StateC>& policy,
                                double gamma) : rewardFunc(rewardFunc), data(data), policy(policy), gamma(gamma)
    {

    }

    virtual void compute(bool computeDerivative = true) = 0;

    inline arma::vec getGradient()
    {
        return gradient;
    }

    inline arma::vec getGradientDiff()
    {
        return dGradient;
    }

    virtual ~NonlinearGradientCalculator()
    {

    }


protected:
    Regressor& rewardFunc;
    Dataset<ActionC,StateC>& data;
    DifferentiablePolicy<ActionC,StateC>& policy;
    double gamma;

    arma::vec gradient;
    arma::mat dGradient;

};

}

#define USE_NONLINEAR_GRADIENT_CALCULATOR_MEMBERS(ActionC, StateC) \
			protected: \
			typedef NonlinearGradientCalculator<ActionC,StateC> Base; \
			using Base::rewardFunc; \
			using Base::data; \
			using Base::policy; \
			using Base::gamma; \
			using Base::gradient; \
			using Base::dGradient; \
			private:


#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENTCALCULATOR_H_ */
