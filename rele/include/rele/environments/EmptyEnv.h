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

#ifndef INCLUDE_RELE_ENVIRONMENTS_EMPTYENV_H_
#define INCLUDE_RELE_ENVIRONMENTS_EMPTYENV_H_

#include "rele/core/ContinuousMDP.h"

namespace ReLe
{

/*!
 * This class implements a MDP with only time as state variable.
 * the reward is zero at each step.
 *
 * This environment can be useful only to test the behavior of
 * a time dependent (or one dimensional state) policy
 */
class EmptyEnv: public ContinuousMDP
{
public:
    /*!
     * Constructor.
     * \param nActions number of actions
     * \param frequency the frequency of the actions
     */
    EmptyEnv(unsigned int nActions, double frequency);

    /*!
     * \see Environment::step
     */
    virtual void step(const DenseAction& action, DenseState& nextState,
                      Reward& reward) override;

    /*!
     * \see Environment::getInitialState
     */
    virtual void getInitialState(DenseState& state) override;


private:
    double dt;

};

}


#endif /* INCLUDE_RELE_ENVIRONMENTS_EMPTYENV_H_ */
