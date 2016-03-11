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

#ifndef INCLUDE_RELE_GENERATORS_FINITEGENERATOR_H_
#define INCLUDE_RELE_GENERATORS_FINITEGENERATOR_H_

#include "rele/core/FiniteMDP.h"

#include <iostream>

namespace ReLe
{

/*!
 * This class contains function to generate finite MDP.
 */
class FiniteGenerator
{
public:
    /*!
     * Return the finite MDP with transition probabilities, reward and
     * reward variance matrices.
     */
    inline FiniteMDP getMDP(double gamma)
    {
        return FiniteMDP(P, R, Rsigma, false, gamma);
    }

    /*!
     * Print transition probabilities, reward and reward variance matrices.
     */
    inline void printMatrices()
    {
        std::cout << "### P matrix ###" << std::endl;
        for (size_t i = 0; i < P.n_rows; i++)
        {
            std::cout << "- action " << i << std::endl;
            arma::mat Pi = P(arma::span(0), arma::span::all, arma::span::all);
            std::cout << Pi << std::endl;
        }

        std::cout << "### R matrix ###" << std::endl;
        for (size_t i = 0; i < R.n_rows; i++)
        {
            std::cout << "- action " << i << std::endl;
            arma::mat Ri = R(arma::span(0), arma::span::all, arma::span::all);
            std::cout << Ri << std::endl;
        }

        std::cout << "### Rsigma matrix ###" << std::endl;
        for (size_t i = 0; i < Rsigma.n_rows; i++)
        {
            std::cout << "- action " << i << std::endl;
            arma::mat Rsigmai = Rsigma(arma::span(0), arma::span::all, arma::span::all);
            std::cout << Rsigmai << std::endl;
        }
    }
protected:
    //! Transition probability matrix
    arma::cube P;
    //! Reward matrix
    arma::cube R;
    //! Reward variance matrix
    arma::cube Rsigma;
    //! Number of states
    size_t stateN;
    //! Number of actions
    unsigned int actionN;
};

}

#endif /* INCLUDE_RELE_GENERATORS_FINITEGENERATOR_H_ */
