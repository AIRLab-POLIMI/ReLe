/*
 * rele,
 *
 *
 * Copyright (C) 2017 Davide Tateo
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


#include "rele/environments/EmptyEnv.h"

namespace ReLe
{

EmptyEnv::EmptyEnv(unsigned int nActions, double frequency) :
    ContinuousMDP(1, nActions, 1, true, true, 0.9, 100), dt(1.0/frequency)
{

}


void EmptyEnv::step(const DenseAction& action, DenseState& nextState,
                    Reward& reward)
{
    currentState(0) += dt;
    currentState.setAbsorbing(false);
    nextState = currentState;
    reward[0] = 0;
}

void EmptyEnv::getInitialState(DenseState& state)
{
    currentState(0) = 0.0;
    currentState.setAbsorbing(false);
    state = currentState;
}


}
