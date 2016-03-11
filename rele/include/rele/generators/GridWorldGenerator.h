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

#ifndef GRIDWORLDGENERATOR_H_
#define GRIDWORLDGENERATOR_H_

#include "FiniteGenerator.h"

namespace ReLe
{

/*!
 * This class contains function to generate a grid world.
 */
class GridWorldGenerator: public FiniteGenerator
{
public:
    /*!
     * Constructor.
     */
    GridWorldGenerator();

    /*!
     * Load a grid world from a text file.
     * \param path path of the text file
     */
    void load(const std::string& path);

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
     * Set reward in case of falling out from the grid world.
     * \param rfall reward in case of falling
     */
    inline void setRfall(double rfall)
    {
        this->rfall = rfall;
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

    /*!
     * Setter.
     * Set the reward obtained at each step.
     * \param rstep reward at each step
     */
    inline void setRstep(double rstep)
    {
        this->rstep = rstep;
    }

private:
    void assignStateNumbers(std::size_t i, std::size_t j);
    void handleChar(std::size_t i, std::size_t j);
    double computeProbability(int currentS, int consideredS, int actionS);
    double computeReward(int consideredS, int actionS);
    int getGoalStateN();
    int getActionState(std::size_t i, std::size_t j, int action);

private:
    //Generator data
    size_t currentState;
    std::vector<std::vector<char>> matrix;
    std::vector<std::vector<int>> stateNMatrix;

    //mdp data
    double p;
    double rgoal;
    double rfall;
    double rstep;

private:
    enum ActionsEnum
    {
        N = 0, S = 1, W = 2, E = 3
    };

};

}
#endif /* GRIDWORLDGENERATOR_H_ */
