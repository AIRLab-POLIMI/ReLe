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

#ifndef AGENT_H_
#define AGENT_H_

#include <vector>

#include "Basics.h"

namespace ReLe
{

class Agent
{
public:
	void initEpisode() = 0;
	void sampleAction(const State& state, Action& action) = 0;
	void step(const Reward& reward, const State& nextState) = 0;
	void endEpisode(const Reward& reward) = 0;
};

}


#endif /* AGENT_H_ */
