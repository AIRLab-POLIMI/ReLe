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

#ifndef MOUNTAINCAR_H_
#define MOUNTAINCAR_H_

#include "DenseMDP.h"

namespace ReLe
{

class MountainCar: public DenseMDP
{
public:
    enum StateLabel
    {
        velocity = 0, position = 1
    };

    enum ConfigurationsLabel
    {
        Sutton, Klein, Random
    };

public:

    MountainCar(ConfigurationsLabel label = Sutton);
    virtual void step(const FiniteAction& action, DenseState& nextState,
                      Reward& reward);
    virtual void getInitialState(DenseState& state);

    ConfigurationsLabel s0type;

};

}

#endif /* MOUNTAINCAR_H_ */
