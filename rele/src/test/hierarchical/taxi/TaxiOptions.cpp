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

bool TaxiLocationOption::canStart(const arma::vec& state)
{
    if(state(sc::x) == location(0) && state(sc::y) == location(1))
        return false;
    else
        return true;

}

double TaxiLocationOption::terminationProbability(const DenseState& state)
{
    if(state(sc::x) == location(0) && state(sc::y) == location(1))
        return 1;
    else
        return 0;
}

void TaxiLocationOption::operator ()(const DenseState& state, FiniteAction& action)
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


/*bool TaxiPickupOption::canStart(const arma::vec& state)
{
    return true;
}

double TaxiPickupOption::terminationProbability(const DenseState& state)
{
    return 1.0;
}

void TaxiPickupOption::operator ()(const DenseState& state, FiniteAction& action)
{
    action.setActionN(an::pickup);
}

bool TaxiDropOffOption::canStart(const arma::vec& state)
{
    return true;
}

double TaxiDropOffOption::terminationProbability(const DenseState& state)
{
    return 1.0;
}

void TaxiDropOffOption::operator ()(const DenseState& state, FiniteAction& action)
{
    action.setActionN(an::dropoff);
}*/

bool TaxiFillUpOption::canStart(const arma::vec& state)
{
    return true;
}

double TaxiFillUpOption::terminationProbability(const DenseState& state)
{
    return 1.0;
}

void TaxiFillUpOption::operator ()(const DenseState& state, FiniteAction& action)
{
    action.setActionN(an::fillup);
}




}
