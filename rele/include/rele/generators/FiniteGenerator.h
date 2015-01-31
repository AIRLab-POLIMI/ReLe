/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#include "FiniteMDP.h"

#include <iostream>

namespace ReLe
{

class FiniteGenerator
{
public:
    inline FiniteMDP getMPD(double gamma)
    {
        return FiniteMDP(P, R, Rsigma, false, gamma);
    }

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
    //output of the algorithm
    arma::cube P;
    arma::cube R;
    arma::cube Rsigma;

    //generator data
    size_t stateN;
    unsigned int actionN;
};

}

#endif /* INCLUDE_RELE_GENERATORS_FINITEGENERATOR_H_ */
