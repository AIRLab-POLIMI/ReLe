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

#include "InternetAds.h"
#include "RandomGenerator.h"

namespace ReLe
{

InternetAds::InternetAds(ExperimentLabel experimentType, double gamma) :
    SimpleMAB(3, gamma),
    experimentType(experimentType)
{
    if(experimentType == First)
        visitors = 100000;
    else
        visitors = 300000;

    cost = 1;
}

void InternetAds::step(const FiniteAction& action, FiniteState& nextState,
                       Reward& reward)
{
    nextState.setStateN(0);
    reward[0] = -cost;

    for(unsigned int i = 0; i < nAds(action); i++)
        for(unsigned int j = 0; j < visitors; j++)
            reward[0] += RandomGenerator::sampleUniformInt(0, 1);
}

unsigned int InternetAds::nAds(const FiniteAction& action)
{
    unsigned int actionN = action.getActionN();

    if(experimentType == First)
    {
        if(actionN == 0)
            return 10;
        else if(actionN == 1)
            return 100;
        else if(actionN == 2)
            return 1000;
    }
    else
    {
        if(actionN == 0)
            return 30;
        else if(actionN == 1)
            return 300;
        else if(actionN == 2)
            return 3000;
    }

    return 0;
}

}
