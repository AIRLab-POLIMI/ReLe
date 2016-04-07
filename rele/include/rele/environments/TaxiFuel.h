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

#include "rele/core/DenseMDP.h"
#include "rele/utils/Range.h"

namespace ReLe
{

#ifndef INCLUDE_RELE_ENVIRONMENTS_TAXIFUEL_H_
#define INCLUDE_RELE_ENVIRONMENTS_TAXIFUEL_H_

/*!
 * This class implements the Taxi Fuel problem.
 * The aim of this problem is to find an optimal
 * policy for a taxi cab in order to let it be able
 * to transport passengers in the desired locations
 * without running out of fuel.
 *
 * References
 * ==========
 * [Dietterich. Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition. JAIR](https://www.jair.org/media/639/live-639-1834-jair.pdf)
 */
//TODO [INTERFACE] not really a dense MDP...
class TaxiFuel: public DenseMDP
{

public:
    /*!
     * Constructor.
     */
    TaxiFuel();

    /*!
     * \see Environment::step
     */
    virtual void step(const FiniteAction& action, DenseState& nextState,
                      Reward& reward) override;

    /*!
     * \see Environment::getInitialState
     */
    virtual void getInitialState(DenseState& state) override;

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

    /*!
     * Return the position of the special locations
     * of the problem.
     * \return vector of positions
     */
    inline std::vector<arma::vec2> getLocations()
    {
        std::vector<arma::vec2> locations;

        locations.push_back(G);
        locations.push_back(Y);
        locations.push_back(B);
        locations.push_back(R);
        locations.push_back(F);

        return locations;
    }



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
