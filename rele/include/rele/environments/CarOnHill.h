/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#ifndef INCLUDE_RELE_ENVIRONMENTS_CARONHILL_H_
#define INCLUDE_RELE_ENVIRONMENTS_CARONHILL_H_

#include "rele/core/DenseMDP.h"

namespace ReLe
{

/*!
 * This class implements the Car On Hill problem.
 * This is a version of mountain car environment, the one proposed by Ernst paper, and is simpler than
 * the original mountain car problem, as the goal can be reached by a random policy.
 *
 * \see MountainCar
 *
 * References
 * ==========
 * [ErnstT, Geurts and Wehrnkel. Tree-Based Batch Mode Reinforcement Learning](http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf)
 */
class CarOnHill: public DenseMDP
{
public:
    enum StateLabel
    {
        position = 0, velocity = 1
    };

public:

    /*!
     * Constructor.
     * \param label configuration type
     */
    CarOnHill(double initialPosition = -0.5,
              double initialVelocity = 0,
              double rewardSigma = 0);

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
    double initialPosition;
    double initialVelocity;
    double rewardSigma;
};

}

#endif /* INCLUDE_RELE_ENVIRONMENTS_CARONHILL_H_ */
