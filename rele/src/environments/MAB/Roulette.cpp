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

Roulette::Roulette(double gamma) :
    MAB<FiniteAction>(gamma)
{
    EnvironmentSettings& task = getWritableSettings();
    task.continuosActionDim = 0;

    // Possible outcome of the roulette
    nOutcomes = 38;
    /*
     * This vector is used to collect the IDs of the possible actions
     * included in a type of bet (e.g. a bet on a single number has 38
     * possible actions if we include the 00 number as in this case).
     */
    actionsId = {38, 96, 111, 133, 134, 145, 151, 157, 158};
    // The squares of the bet
    nSquares = {1, 2, 3, 4, 5, 6, 12, 18, 0};

    task.finiteActionDim = actionsId(actionsId.n_elem - 1);

    // The amount to bet
    bet = 1;
}

void Roulette::step(const FiniteAction& action, FiniteState& nextState, Reward& reward)
{
    // As a MAB, each action returns to the current state
    nextState.setStateN(0);
    // The reward is the difference between the bet and the outcome
    reward[0] = -bet + computeReward(action);
}

double Roulette::computeReward(const FiniteAction& action)
{
    unsigned int actionN = action.getActionN();

    // If the bet is on a single number, nSquares = 1
    if(actionN < actionsId(0))
        return rouletteReward(nSquares(0));
    // Check which is the type of bet of the action
    for(unsigned int i = 1; i < nSquares.n_elem - 1; i++)
        if(actionN >= actionsId(i) && actionN < actionsId(i + 1))
            return rouletteReward(nSquares(i));
    // The action is to leave the table. So the total reward is 0 (bet is summed to -bet)
    return bet;
}

double Roulette::rouletteReward(double nSquares)
{
    // Probability to win
    double p = nSquares / nOutcomes;

    // Sample the event with its probability to occur
    if(RandomGenerator::sampleEvent(p))
    {
        // In case of win

        // Payout formula
        unsigned int payout = floor(36 / nSquares) - 1;
        // The original bet summed to the bet times the payout is the reward
        return bet + bet * payout;
    }
    // Lost the bet: the reward is -bet
    return 0;
}

}
