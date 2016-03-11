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

#include "rele/core/DenseMDP.h"

namespace ReLe
{

/*!
 * This class implements the Deep Sea Treasure problem.
 * This task is a grid world modeling a submarine
 * environment with multiple treasures with different
 * values. The aim is to minimize the time to reach
 * the treasures and maximize the values of reached
 * treasures.
 * For further information see <a href="https://books.google.it/books?id=iIlrCQAAQBAJ&pg=PA375&lpg=PA375&dq=deep+sea+treasure+problem+ai&source=bl&ots=jB5wkUj9t_&sig=YUkqRATglHxq5dU0PGv1u_ePJ2k&hl=it&sa=X&ved=0ahUKEwiY-bf8z7TLAhUrM5oKHdd_C5AQ6AEIHDAA#v=onepage&q=deep%20sea%20treasure%20problem%20ai&f=false">here</a>.
 */
class DeepSeaTreasure: public DenseMDP
{
public:
    /*!
     * Constructor.
     */
    DeepSeaTreasure();

    /*!
     * \see Environment::step
     */
    virtual void step(const FiniteAction& action, DenseState& nextState,
                      Reward& reward) override;

    /*!
     * \see Environment::getInitialState
     */
    virtual void getInitialState(DenseState& state) override;

private:
    double deep_reward_treasure(DenseState& state);
    bool deep_check_black(int x, int y);

private:
    unsigned int xdim, ydim;
};

}

#endif /* DEEPSEATREASURE_H_ */
