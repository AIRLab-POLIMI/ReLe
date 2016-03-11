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

#ifndef INCLUDE_RELE_GENERATORS_SIMPLECHAINGENERATOR_H_
#define INCLUDE_RELE_GENERATORS_SIMPLECHAINGENERATOR_H_

#include "FiniteGenerator.h"

namespace ReLe
{

/*!
 * This class contains function to generate a simple Markov chain.
 */
class SimpleChainGenerator: public FiniteGenerator
{
public:
    /*!
     * Constructor.
     */
    SimpleChainGenerator();

    /*!
     * Initialize the Markov chain.
     * \param size the size of the Markov chain
     * \param goalState goal state index
     */
    void generate(std::size_t size, std::size_t goalState);

    /*!
     * Setter.
     * Set probability of success of actions.
     * \param p probability of success of actions
     */
    inline void setP(double p)
    {
        this->p = p;
    }

    /*!
     * Setter.
     * Set reward in case of reaching goal state.
     * \param rgoal reward when reaching goal state
     */
    inline void setRgoal(double rgoal)
    {
        this->rgoal = rgoal;
    }

private:

    inline bool isLeftmostState(std::size_t state)
    {
        return state == 0;
    }

    inline bool isRightmostState(std::size_t state)
    {
        return state == stateN - 1;
    }

    inline std::size_t previousState(std::size_t state)
    {
        return state - 1;
    }

    inline std::size_t nextState(std::size_t state)
    {
        return state + 1;
    }

    void computeReward(size_t goalState);
    void computeprobabilities();

private:
    //mdp data
    double p;
    double rgoal;

private:
    enum ActionsEnum
    {
        RIGHT = 0, LEFT = 1
    };

};
}

#endif /* INCLUDE_RELE_GENERATORS_SIMPLECHAINGENERATOR_H_ */
