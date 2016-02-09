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

namespace ReLe
{

template<class ActionC, class StateC>
class GradientCalculator
{
public:
    GradientCalculator(BasisFunctions& basis,
                       Dataset<ActionC,StateC>& data,
                       DifferentiablePolicy<ActionC,StateC>& policy,
                       double gamma):
        basis(basis), data(data), policy(policy), gamma(gamma),
        gradientDiff(policy.getParametersSize(), basis.size(), arma::fill::zeros)
    {
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
    virtual arma::vec computeGradientFeature(BasisFunction& basis) = 0;

private:
    virtual void computeGradientDiff()
    {
    	if(!computed)
    	{
			unsigned int parametersN = policy.getParametersSize();
			unsigned int featuresN = basis.size();

			for(int i = 0; i < basis.size(); i++)
			{
				gradientDiff.col(i) = computeGradientFeature(*basis[i]);
			}

			computed = true;
    	}
    }

protected:
    BasisFunctions& basis;
    Dataset<ActionC,StateC>& data;
    DifferentiablePolicy<ActionC,StateC>& policy;
    double gamma;

private:
    bool computed;
    arma::mat gradientDiff;

};


}

#define USE_GRADIENT_CALCULATOR_MEMBERS(ActionC, StateC) \
			typedef GradientCalculator<ActionC,StateC> Base; \
			using Base::basis; \
			using Base::data; \
			using Base::policy; \
			using Base::gamma;


#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENTCALCULATOR_H_ */
