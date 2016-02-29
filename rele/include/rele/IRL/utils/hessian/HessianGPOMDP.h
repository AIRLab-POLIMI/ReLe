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

#ifndef INCLUDE_RELE_IRL_UTILS_HESSIAN_HESSIANGPOMDP_H_
#define INCLUDE_RELE_IRL_UTILS_HESSIAN_HESSIANGPOMDP_H_

#include "rele/IRL/utils/hessian/HessianCalculator.h"

namespace ReLe
{
template<class ActionC, class StateC>
class HessianGPOMDP : public HessianCalculator<ActionC, StateC>
{
protected:
    USE_HESSIAN_CALCULATOR_MEMBERS(ActionC, StateC)

public:
    HessianGPOMDP(Features& phi,
                  Dataset<ActionC,StateC>& data,
                  DifferentiablePolicy<ActionC,StateC>& policy,
                  double gamma) : HessianCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {
    	for(auto& episode : data)
    	{

    		for(auto& tr : episode)
    		{

    		}

    	}
    }

    virtual ~HessianGPOMDP()
    {

    }

protected:
    virtual arma::cube computeHessianDiff() override
    {
        //TODO implement
        return arma::cube();
    }



};

template<class ActionC, class StateC>
class HessianGPOMDPBase : public HessianCalculator<ActionC, StateC>
{
protected:
    USE_HESSIAN_CALCULATOR_MEMBERS(ActionC, StateC)

public:
    HessianGPOMDPBase(Features& phi,
                      Dataset<ActionC,StateC>& data,
                      DifferentiablePolicy<ActionC,StateC>& policy,
                      double gamma) : HessianCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~HessianGPOMDPBase()
    {

    }

protected:
    virtual arma::cube computeHessianDiff() override
    {
        //TODO implement
        return arma::cube();
    }
};

}


#endif /* INCLUDE_RELE_IRL_UTILS_HESSIAN_HESSIANGPOMDP_H_ */
