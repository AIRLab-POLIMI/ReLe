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

#include "Roulette.h"
#include "RandomGenerator.h"

namespace ReLe
{

Roulette::Roulette(ExperimentLabel rouletteType, double gamma) :
		LinearMAB<FiniteAction>(gamma),
		rouletteType(rouletteType)
{
	EnvironmentSettings& task = getWritableSettings();
	task.continuosActionDim = 0;
	if(rouletteType == American)
	{
		nNumbers = 38;
		nBets = 171;
	}
	else
	{
		nNumbers = 37;
		nBets = 150;
	}
	task.finiteActionDim = nBets;

    bet = 1;
}

void Roulette::step(const FiniteAction& action, FiniteState& nextState, Reward& reward)
{
	nextState.setStateN(0);

	if(action.getActionN() == nBets - 1)
		nextState.setAbsorbing();
	reward[0] = computeReward(action);
}

double Roulette::computeReward(const FiniteAction& action)
{
	unsigned int nSquares;
	double payout;
	double p;

	unsigned int actionN = action.getActionN();
	if(rouletteType == American)
	{
		if(actionN >= 0 && actionN <= 39)
			nSquares = 1;
		else if(actionN >= 40 && actionN <= 97)
			nSquares = 2;
		else if(actionN >= 98 && actionN <= 109)
			nSquares = 3;
		else if(actionN >= 110 && actionN <= 121)
			nSquares = 4;
		else if(actionN == 122)
			nSquares = 5;
		else if(actionN >= 123 && actionN <= 133)
			nSquares = 6;
		else if(actionN >= 134 && actionN <= 139)
			nSquares = 12;
		else if(actionN >= 140 && actionN <= 145)
			nSquares = 18;
		else if(actionN == 170)
			return 0;
	}
	else
	{
		if(actionN >= 0 && actionN <= 37)
			nSquares = 2;
		else if(actionN >= 40 && actionN <= 95)
		{

		}
		else if(actionN == 169)
			return 0;
	}

	return rouletteReward(nSquares);
}

double Roulette::rouletteReward(unsigned int nSquares)
{
	double payout = 36 / nSquares - 1; // Payout formula
	double p = nSquares / nNumbers; // Probability to win

	if(RandomGenerator::sampleEvent(p))
		return bet + bet * payout;
	return -bet;
}

}
