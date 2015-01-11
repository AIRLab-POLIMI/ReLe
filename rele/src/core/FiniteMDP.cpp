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

FiniteMDP::FiniteMDP(double*** Pdata, double*** Rdata,
			std::size_t statesNumber, std::size_t actionsNumber) : Envirorment()
{
	initP(actionsNumber, statesNumber, Pdata);
	initR(actionsNumber, statesNumber, Rdata);
}

bool FiniteMDP::step(const FiniteAction& action, FiniteState& nextState,
			Reward& reward)
{

	//Compute next state
	int u = action.getActionN();
	size_t x = currentState.getStateN();
	int xn = RandomGenerator::sampleDiscrete(P[u][x]);

	currentState.setStateN(xn);
	nextState.setStateN(xn);

	//compute reward
	double m = R[u][xn][0];
	double sigma = R[u][xn][1];
	double r = RandomGenerator::sampleNormal(m, sigma);

	reward.push_back(r);

	return false; //TODO change this

}

void FiniteMDP::getInitialState(FiniteState& state)
{
	int x = RandomGenerator::sampleUniformInt(0, P[0].size());

	currentState.setStateN(x);
	state.setStateN(x);
}

void FiniteMDP::initP(std::size_t actionsNumber, std::size_t statesNumber,
			double*** Pdata)
{
	P.resize(actionsNumber);
	for (size_t k = 0; k < actionsNumber; k++)
	{
		P[k].resize(statesNumber);
		for (size_t i = 0; i < statesNumber; i++)
		{
			P[k][i].resize(statesNumber);
			for (size_t j = 0; j < statesNumber; j++)
			{
				P[k][i][j] = Pdata[k][i][j];
			}
		}
	}
}

void FiniteMDP::initR(std::size_t actionsNumber, std::size_t statesNumber,
			double*** Rdata)
{
	R.resize(actionsNumber);
	for (size_t k = 0; k < actionsNumber; k++)
	{
		R[k].resize(statesNumber);
		for (size_t i = 0; i < statesNumber; i++)
		{
			R[k][i].resize(2);
			R[k][i][0] = Rdata[k][i][0]; //mean
			R[k][i][1] = Rdata[k][i][1]; //variance
		}
	}
}

}
