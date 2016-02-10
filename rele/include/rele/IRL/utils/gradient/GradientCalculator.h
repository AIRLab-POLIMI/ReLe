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
class GradientCalculator
{
public:
    GradientCalculator(Features& phi,
                       Dataset<ActionC,StateC>& data,
                       DifferentiablePolicy<ActionC,StateC>& policy,
                       double gamma):
        phi(phi), data(data), policy(policy), gamma(gamma),
        gradientDiff(policy.getParametersSize(), phi.rows(), arma::fill::zeros)
    {
        assert(phi.cols() == 1);
        computed = false;
    }

    arma::vec computeGradient(const arma::vec& w)
    {
        computeGradientDiff();

        return gradientDiff*w;
    }

    arma::mat getGradientDiff()
    {
        computeGradientDiff();

        return gradientDiff;
    }

    virtual ~GradientCalculator()
    {

    }


protected:
    virtual arma::mat computeGradientDiff() = 0;

protected:
    Features& phi;
    Dataset<ActionC,StateC>& data;
    DifferentiablePolicy<ActionC,StateC>& policy;
    double gamma;

private:
    bool computed;
    arma::mat gradientDiff;

};


}

#define USE_GRADIENT_CALCULATOR_MEMBERS(ActionC, StateC) \
			protected: \
			typedef GradientCalculator<ActionC,StateC> Base; \
			using Base::phi; \
			using Base::data; \
			using Base::policy; \
			using Base::gamma; \
			private:


#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENTCALCULATOR_H_ */
