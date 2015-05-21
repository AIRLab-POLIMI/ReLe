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

#include "DenseMDP.h"
#include "Range.h"

namespace ReLe
{

#ifndef INCLUDE_RELE_ENVIRONMENTS_TAXIFUEL_H_
#define INCLUDE_RELE_ENVIRONMENTS_TAXIFUEL_H_


//TODO not really a dense MDP...
class TaxiFuel: public DenseMDP
{

public:
    TaxiFuel();
    virtual void step(const FiniteAction& action, DenseState& nextState,
                      Reward& reward);
    virtual void getInitialState(DenseState& state);

    enum StateComponents
    {
        //taxi position
        x = 0,
        y,

        //taxi state
        fuel,
        onBoard,

        //passenger
        location,
        destination,
        //state size
        STATESIZE
    };

    enum ActionNames
    {
        up = 0,
        down,
        left,
        right,
        pickup,
        dropoff,
        fillup,

        //action N
        ACTIONNUMBER
    };

private:
    bool atLocation();
    bool atDestination();
    bool atFuelStation();
	arma::vec2 extractTarget(int targetN);

private:
    arma::vec2 G;
    arma::vec2 Y;
    arma::vec2 B;
    arma::vec2 R;
    arma::vec2 F;

    Range gridDim;
};


}

#endif /* INCLUDE_RELE_ENVIRONMENTS_TAXIFUEL_H_ */
