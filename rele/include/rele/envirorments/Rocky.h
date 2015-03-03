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

#ifndef INCLUDE_RELE_ENVIRORMENTS_ROCKY_H_
#define INCLUDE_RELE_ENVIRORMENTS_ROCKY_H_

#include "ContinuousMDP.h"

#include <vector>

namespace ReLe
{

class Rocky : public ContinuousMDP
{
public:
    Rocky();
    virtual void step(const DenseAction& action, DenseState& nextState,
                      Reward& reward);
    virtual void getInitialState(DenseState& state);

private:
    enum StateComponents
    {
        //robot state
        x = 0,
        y,
        theta,
        //robot sensors
        energy,
        food,
        //rocky state
        xr,
        yr,
        thetar,
        //state size
        STATESIZE
    };

private:
    std::vector<arma::vec2> foodSpots;
    const double dt;
    const double maxOmega;
    const double maxV;

private:


};


}

#endif /* INCLUDE_RELE_ENVIRORMENTS_ROCKY_H_ */
