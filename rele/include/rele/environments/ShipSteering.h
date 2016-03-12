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

#ifndef INCLUDE_RELE_ENVIRONMENTS_SHIPSTEERING_H_
#define INCLUDE_RELE_ENVIRONMENTS_SHIPSTEERING_H_

#include "rele/core/ContinuousMDP.h"
#include "rele/utils/Range.h"

namespace ReLe
{

/*!
 * This class implements the Ship Steering problem.
 * The aim of this problem is to let a ship pass through
 * a gate when starting from a random position and moving
 * at constant speed.
 * For further information see <a href="http://people.cs.umass.edu/~mahadeva/papers/icml03-1.pdf">here</a>.
 *
 * References
 * ==========
 * [Ghavamzadeh, Mahadevan. Hierarchical Policy Gradient Algorithms. ICML 2013](http://people.cs.umass.edu/~mahadeva/papers/icml03-1.pdf)
 */
class ShipSteering : public ContinuousMDP
{
public:
    /*!
     * Constructor.
     * \param small field size
     */
    ShipSteering(bool small = true);

    /*!
     * \see Environment::step
     */
    virtual void step(const DenseAction& action, DenseState& nextState,
                      Reward& reward) override;

    /*!
     * \see Environment::getInitialState
     */
    virtual void getInitialState(DenseState& state) override;

    enum StateComponents
    {
        // ship state
        x = 0,
        y,
        theta,
        omega,
        // state size
        STATESIZE
    };

private:
    bool throughGate(const arma::vec& start, const arma::vec& end);
    double cross2D(const arma::vec& v, const arma::vec& w);


private:
    arma::vec2 gateS;
    arma::vec2 gateE;

    Range rangeField;
    Range rangeOmega;

    static constexpr double v = 3.0;
    static constexpr double dt = 0.2;
    static constexpr double T = 5.0;
};

}



#endif /* INCLUDE_RELE_ENVIRONMENTS_SHIPSTEERING_H_ */
