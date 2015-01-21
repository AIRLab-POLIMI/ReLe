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

#ifndef FINITEMDP_H_
#define FINITEMDP_H_

#include "Basics.h"
#include "Envirorment.h"

#include <vector>

#include <armadillo>

namespace ReLe
{

class FiniteMDP: public Envirorment<FiniteAction, FiniteState>
{
public:
    FiniteMDP(arma::cube P, arma::cube R, bool isFiniteHorizon, double gamma =
                  1.0, unsigned int horizon = 0);

    virtual void step(const FiniteAction& action, FiniteState& nextState,
                      Reward& reward);
    virtual void getInitialState(FiniteState& state);

private:
    arma::cube P;
    arma::cube R;
    FiniteState currentState;

    void chekMatricesDimensions(const arma::cube& P, const arma::cube& R);
    void setupEnvirorment(bool isFiniteHorizon, unsigned int horizon,
                          double gamma, const arma::cube& P);
};

}

#endif /* FINITEMDP_H_ */
