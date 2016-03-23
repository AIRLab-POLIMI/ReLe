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

#ifndef DENSEMDP_H_
#define DENSEMDP_H_

#include "rele/core/Environment.h"

namespace ReLe
{

/*!
 * This class is the abstract interface for all MDPs that have continuous state space and finite actions.
 */
class DenseMDP: public Environment<FiniteAction, DenseState>
{
public:
    /*!
     * Constructor.
     * \param settings the pointer to the environment settings
     */
    DenseMDP(EnvironmentSettings* settings);

    /*!
     * Constructor.
     * \param stateSize the dimensionality of the state space
     * \param actionN the number of actions
     * \param rewardSize the dimensionality of the reward function
     * \param isFiniteHorizon if the MDP has finite horizon
     * \param isEpisodic if the task is episodic
     * \param gamma the MDP discount factor
     * \param horizon the MDP horizon
     */
    DenseMDP(std::size_t stateSize, unsigned int actionN, size_t rewardSize, bool isFiniteHorizon,
             bool isEpisodic, double gamma = 1.0, unsigned int horizon = 0);

    /*!
     * Destructor.
     */
    virtual ~DenseMDP()
    {}

protected:
    void setupenvironment(std::size_t stateSize, unsigned int actionN, size_t rewardSize,
                          bool isFiniteHorizon, bool isEpisodic, unsigned int horizon,
                          double gamma);

protected:
    //! the MDP current state
    DenseState currentState;

};

}

#endif /* DENSEMDP_H_ */
