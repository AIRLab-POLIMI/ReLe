/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef FINITEMDP_H_
#define FINITEMDP_H_

#include "Basics.h"
#include "Envirorment.h"

#include <vector>

namespace ReLe
{

class FiniteMDP : public Envirorment<FiniteAction, FiniteState>
{

public:
	FiniteMDP(double*** Pdata, double*** Rdata, std::size_t statesNumber, std::size_t actionsNumber);
	virtual bool step(const FiniteAction& action, FiniteState& nextState,
					Reward& reward);
	virtual void getInitialState(FiniteState& state);


private:
	std::vector<std::vector<std::vector<double>>> P;
	std::vector<std::vector<std::vector<double>>> R;
	FiniteState currentState;

	void initP(std::size_t actionsNumber, std::size_t statesNumber, double*** Pdata);
	void initR(std::size_t actionsNumber, std::size_t statesNumber, double*** Rdata);
};

}


#endif /* FINITEMDP_H_ */
