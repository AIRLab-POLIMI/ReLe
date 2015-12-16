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

/*
 * Written by: Carlo D'Eramo
 */

#include "MAB/Roulette.h"
#include "RandomGenerator.h"

namespace ReLe
{

Roulette::Roulette(ExperimentLabel rouletteType, double gamma) :
    MAB<FiniteAction>(gamma),
    rouletteType(rouletteType)
{
    EnvironmentSettings& task = getWritableSettings();
    task.continuosActionDim = 0;

    if(rouletteType == American)
    {
    	nOutcomes = 38;
    	actionsId = {38, 96, 111, 133, 134, 145, 151, 157, 158};
    	nSquares = {1, 2, 3, 4, 5, 6, 12, 18, 0};
    }
    else
    {
        nOutcomes = 37;
        actionsId = {37, 58, 15, 22, 11, 6, 6, 1};
        nSquares = {1, 2, 3, 4, 6, 12, 18, 0};
    }
    task.finiteActionDim = actionsId(actionsId.n_elem - 1);

    bet = 1;
}

void Roulette::step(const FiniteAction& action, FiniteState& nextState, Reward& reward)
{
    nextState.setStateN(0);
    reward[0] = -bet + computeReward(action);

    const EnvironmentSettings task = getSettings();
    if(action.getActionN() == task.finiteActionDim - 1)
    	nextState.setAbsorbing();
}

double Roulette::computeReward(const FiniteAction& action)
{
	unsigned int actionN = action.getActionN();

	if(actionN < actionsId(0))
		return rouletteReward(nSquares(0));
	for(unsigned int i = 1; i < nSquares.n_elem - 1; i++)
		if(actionN >= actionsId(i) && actionN < actionsId(i + 1))
			return rouletteReward(nSquares(i));
	return bet;
}

double Roulette::rouletteReward(double nSquares)
{
    double p = nSquares / nOutcomes; // Probability to win

    if(RandomGenerator::sampleEvent(p))
    {
        unsigned int payout = floor(36 / nSquares) - 1; // Payout formula
        return bet + bet * payout;
    }
    return 0;
}

}
