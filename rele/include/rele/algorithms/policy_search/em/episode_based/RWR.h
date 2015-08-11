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

#ifndef INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_EM_EPISODE_BASED_RWR_H_
#define INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_EM_EPISODE_BASED_RWR_H_

#include "policy_search/BlackBoxAlgorithm.h"

namespace ReLe
{

template<class ActionC, class StateC>
class RWR: public BlackBoxAlgorithm<ActionC, StateC, ParametricNormal, BlackBoxOutputData>
{
	USE_BBA_MEMBERS(BlackBoxOutputData)

public:
	RWR()
	{

	}

	virtual ~RWR()
	{

	}

protected:
    virtual void init()
    {

    }

    virtual void afterPolicyEstimate()
    {

    }

    virtual void afterMetaParamsEstimate()
    {

    }

};

}

#endif /* INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_EM_EPISODE_BASED_RWR_H_ */
