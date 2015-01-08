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

#ifndef ENVIRORMENT_H_
#define ENVIRORMENT_H_

#include <vector>

#include "Basics.h"

namespace ReLe
{

struct EnvirormentSettings
{
	double gamma;
	bool isDiscreteActions;
	bool isDiscreteStates;
	bool isAverageReward;
	bool isFiniteHorizon;
	bool isEpisodic;
	int horizon;
};

class Envirorment
{
public:
	bool step(const Action& action, State& nextState, Reward reward) = 0;
	void getInitialState(State& state) = 0;
};

}

#endif /* ENVIRORMENT_H_ */
