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

#ifndef INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_GRADIENT_HIERARCHICAL_HIERARCHICALGRADIENTOUTPUTDATA_H_
#define INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_GRADIENT_HIERARCHICAL_HIERARCHICALGRADIENTOUTPUTDATA_H_

#include "HierarchicalOutputData.h"
#include "policy_search/gradient/onpolicy/GradientOutputData.h"

namespace ReLe
{

class HierarchicalGradientOutputData : public HierarchicalOutputData, public GradientIndividual
{
public:
	HierarchicalGradientOutputData() : HierarchicalOutputData(), GradientIndividual()
	{

	}

    virtual void writeData(std::ostream& os)
    {
        GradientIndividual::writeData(os);
        //HierarchicalOutputData::writeData(os);
    }

    virtual void writeDecoratedData(std::ostream& os)
    {
        this->writeData(os);
    }
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_GRADIENT_HIERARCHICAL_HIERARCHICALGRADIENTOUTPUTDATA_H_ */
