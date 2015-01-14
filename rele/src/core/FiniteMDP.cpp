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

#include "FiniteMDP.h"
#include "RandomGenerator.h"

using namespace std;

namespace ReLe
{

void FiniteMDP::step(const FiniteAction& action, FiniteState& nextState,
			Reward& reward)
{

	//Compute next state
	unsigned int u = action.getActionN();
	size_t x = currentState.getStateN();
	size_t xn = RandomGenerator::sampleDiscrete(P[u][x]);

	currentState.setStateN(xn);
	nextState.setStateN(xn);

	//compute reward
	double m = R[xn][0];
	double sigma = R[xn][1];
	double r = RandomGenerator::sampleNormal(m, sigma);

	reward.push_back(r);

}

void FiniteMDP::getInitialState(FiniteState& state)
{
	size_t x = RandomGenerator::sampleUniformInt(0, P[0].size() - 1);

	currentState.setStateN(x);
	state.setStateN(x);
}

}
