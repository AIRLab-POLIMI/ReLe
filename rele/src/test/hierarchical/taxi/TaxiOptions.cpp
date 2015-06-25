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

#include "TaxiOptions.h"
#include "TaxiFuel.h"

namespace ReLe
{

using sc=TaxiFuel::StateComponents;
using an=TaxiFuel::ActionNames;

TaxiLocationOption::TaxiLocationOption(arma::vec location) : location(location)
{

}

void TaxiLocationOption::operator ()(const DenseState& state, FiniteAction& action)
{
    goToLocation(state, action);
}

void TaxiLocationOption::goToLocation(const DenseState& state, FiniteAction& action)
{
    if(state(sc::x) > location(0))
        action.setActionN(an::left);
    else if(state(sc::x) < location(0))
        action.setActionN(an::right);
    else if(state(sc::y) > location(1))
        action.setActionN(an::down);
    else
        action.setActionN(an::up);
}


bool TaxiLocationOption::canStart(const arma::vec& state)
{
    return state(sc::x) != location(0) || state(sc::y) != location(1);
}

double TaxiLocationOption::terminationProbability(const DenseState& state)
{
    if (state(sc::x) == location(0) && state(sc::y) == location(1))
        return 1.0;
    else
        return 0.0;
}

bool TaxiSimpleOption::canStart(const arma::vec& state)
{
    return true;
}

double TaxiSimpleOption::terminationProbability(const DenseState& state)
{
    return 1.0;
}

void TaxiPickupOption::operator ()(const DenseState& state, FiniteAction& action)
{
    action.setActionN(an::pickup);
}

void TaxiDropOffOption::operator ()(const DenseState& state, FiniteAction& action)
{
    action.setActionN(an::dropoff);
}

void TaxiFillUpOption::operator ()(const DenseState& state, FiniteAction& action)
{
    action.setActionN(an::fillup);
}


/*
 * Complex Options
 */

TaxiComplexOption::TaxiComplexOption(std::vector<arma::vec2>& locations, ActionType action)
    : actionType(action), locations(locations)
{

}

bool TaxiComplexOption::canStart(const arma::vec& state)
{
    return actionType != DropOff || state(sc::onBoard) == 1;
}

void TaxiComplexOption::operator ()(const DenseState& state, FiniteAction& action)
{
    auto& location = getLocation(state);
    if(state(sc::x) != location(0) || state(sc::y) != location(1))
    {
        goToLocation(state, action);
    }
    else
    {
        switch(actionType)
        {
        case PickUp:
            action.setActionN(an::pickup);
            break;

        case DropOff:
            action.setActionN(an::dropoff);
            break;

        case FillUp:
            action.setActionN(an::fillup);
            break;

        }


    }
}

arma::vec& TaxiComplexOption::getLocation(const DenseState& state)
{
    switch(actionType)
    {
    case PickUp:
        return locations[state(sc::location)];

    case DropOff:
        return locations[state(sc::destination)];

    case FillUp:
        return locations.back();

    }

}

void TaxiComplexOption::goToLocation(const DenseState& state, FiniteAction& action)
{
    auto& location = getLocation(state);
    if(state(sc::x) > location(0))
        action.setActionN(an::left);
    else if(state(sc::x) < location(0))
        action.setActionN(an::right);
    else if(state(sc::y) > location(1))
        action.setActionN(an::down);
    else
        action.setActionN(an::up);
}


TaxiComplexPickupOption::TaxiComplexPickupOption(std::vector<arma::vec2>& location)
    : TaxiComplexOption(location, ActionType::PickUp)
{

}

double TaxiComplexPickupOption::terminationProbability(const DenseState& state)
{
    if(state(sc::onBoard) == 1)
        return 1;
    else
        return 0;
}


TaxiComplexDropOffOption::TaxiComplexDropOffOption(std::vector<arma::vec2>& location)
    : TaxiComplexOption(location, ActionType::DropOff)
{

}

double TaxiComplexDropOffOption::terminationProbability(const DenseState& state)
{
    if(state(sc::onBoard) == -1)
        return 1;
    else
        return 0;
}

TaxiComplexFillupOption::TaxiComplexFillupOption(std::vector<arma::vec2>& location)
	: TaxiComplexOption(location, ActionType::FillUp)
{

}

double TaxiComplexFillupOption::terminationProbability(const DenseState& state)
{
	if (state[sc::fuel] == 12)
		return 1;
	else
		return 0;
}




}
