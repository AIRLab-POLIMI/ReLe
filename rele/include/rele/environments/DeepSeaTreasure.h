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

#ifndef DEEPSEATREASURE_H_
#define DEEPSEATREASURE_H_

#include "DenseMDP.h"

/**
 * Environment designed according to
 *
 * TODO
 */

namespace ReLe
{

class DeepSeaTreasure: public DenseMDP
{
public:

    DeepSeaTreasure();

    virtual void step(const FiniteAction& action, DenseState& nextState,
                      Reward& reward) override;
    virtual void getInitialState(DenseState& state) override;

private:
    double deep_reward_treasure(DenseState& state);
    bool deep_check_black(int x, int y);

private:
    unsigned int xdim, ydim;
};

}

#endif /* DEEPSEATREASURE_H_ */
