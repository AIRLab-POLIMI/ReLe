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

#include "rele/environments/DeepSeaTreasure.h"

#include "rele/utils/RandomGenerator.h"
#include <cassert>

using namespace std;

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// DEEP ENVIRONMENTS
///////////////////////////////////////////////////////////////////////////////////////

DeepSeaTreasure::DeepSeaTreasure()
    : DenseMDP(2,4,2,false,true,1.0,50), xdim(11), ydim(10)
{
    currentState.set_size(2);
}


void DeepSeaTreasure::step(const FiniteAction& action, DenseState& nextState, Reward& reward)
{
    unsigned int a = action.getActionN();
    int i = currentState[0], j = currentState[1];
    int j1, j2, i3, i4;

    switch (a)
    {
    case 0:
        // left
        j1 = max(1,j-1);
        if (!deep_check_black(i,j1))
            j1 = j;
        currentState[0] = i;
        currentState[1] = j1;
        break;
    case 1:
        // right
        j2 = min(static_cast<int>(ydim),j+1);
        if (!deep_check_black(i,j2))
            j2 = j;
        currentState[0] = i;
        currentState[1] = j2;
        break;
    case 2:
        // up
        i3 = max(1,i-1);
        if (!deep_check_black(i3,j))
            i3 = i;
        currentState[0] = i3;
        currentState[1] = j;
        break;
    case 3:
        // down
        i4 = min(static_cast<int>(xdim),i+1);
        if (!deep_check_black(i4,j))
            i4 = i;
        currentState[0] = i4;
        currentState[1] = j;
        break;
    default:
        cerr << "DEEPSEATR: Unknown action" << endl;
        abort();
        break;
    }


    // treasure value
    double reward1 = deep_reward_treasure(currentState);
    // time
    double reward2 = -1;
    if (reward1 == 0)
        currentState.setAbsorbing(false);
    else
        currentState.setAbsorbing(true);

    assert(!std::isnan(reward1));
    assert(!std::isnan(reward2));
    assert(!std::isinf(reward1));
    assert(!std::isinf(reward2));
    reward[0] = reward1;
    reward[1] = reward2;
    nextState = currentState;
}

void DeepSeaTreasure::getInitialState(DenseState& state)
{
    currentState.setAbsorbing(false);
    currentState[0] = 1;
    currentState[1] = 1;
    state = currentState;
}

double DeepSeaTreasure::deep_reward_treasure(DenseState& state)
{
    arma::mat reward(xdim+1,ydim+1,arma::fill::zeros);
    reward(2,1) = 1;
    reward(3,2) = 2;
    reward(4,3) = 3;
    reward(5,4) = 5;
    reward(5,5) = 8;
    reward(5,6) = 16;
    reward(8,7) = 24;
    reward(8,8) = 50;
    reward(10,9) = 74;
    reward(11,10) = 124;
    return reward(state[0],state[1]);

}

bool DeepSeaTreasure::deep_check_black(int x, int y)
{

    int i, j;

    for (i = 3; i < xdim; ++i)
    {
        for (j = 1; j < i - 2; ++j)
        {
            if ((x == i) && (y == j))
            {
                return false;
            }
        }
    }

    if ((x == 6 && y == 5) || (x == 6 && y == 6) || (x == 7 && y == 6) || (x == 9 && y == 8))
        return false;
    return true;
}

}  //end namespace

