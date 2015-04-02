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

#ifndef INCLUDE_RELE_DYNAMIC_PROGRAMMING_BASICDYNAMICPROGRAMMING_H_
#define INCLUDE_RELE_DYNAMIC_PROGRAMMING_BASICDYNAMICPROGRAMMING_H_

#include "Solver.h"
#include "FiniteMDP.h"

#include <armadillo>

namespace ReLe
{


class DynamicProgrammingAlgorithm : Solver<FiniteAction, FiniteState>
{
public:
    DynamicProgrammingAlgorithm(FiniteMDP& mdp);

protected:
    FiniteMDP& mdp;
    size_t stateN;
    unsigned int actionN;
    double gamma;

    const arma::cube& P;
    const arma::cube& R;
};

class PolicyIteration : public DynamicProgrammingAlgorithm
{
public:
    PolicyIteration(FiniteMDP& mdp);

    virtual void solve();
    virtual Dataset<FiniteAction, FiniteState> test();
    virtual std::string printPolicy();

    virtual ~PolicyIteration();

};

class ValueIteration : public DynamicProgrammingAlgorithm
{
public:
    ValueIteration(FiniteMDP& mdp, double eps);

    virtual void solve();
    virtual Dataset<FiniteAction, FiniteState> test();
    virtual std::string printPolicy();

    virtual ~ValueIteration();


private:
    double eps;

    arma::vec Vold;
    arma::vec V;
};



}


#endif /* INCLUDE_RELE_DYNAMIC_PROGRAMMING_BASICDYNAMICPROGRAMMING_H_ */
