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

/*
 * Written by: Carlo D'Eramo
 */

#ifndef INCLUDE_RELE_ENVIRONMENTS_MAB_SIMPLEMAB_H_
#define INCLUDE_RELE_ENVIRONMENTS_MAB_SIMPLEMAB_H_

#include "rele_mab/environments/MAB.h"


namespace ReLe
{

class DiscreteMAB: public MAB<FiniteAction>
{

public:

    DiscreteMAB(arma::vec P, arma::vec R, unsigned int horizon = 1);
    DiscreteMAB(arma::vec P, double r = 1, unsigned int horizon = 1);


    DiscreteMAB(unsigned int nArms, double r, unsigned int horizon = 1);
    DiscreteMAB(unsigned int nArms, unsigned int horizon = 1);
    arma::vec getP();
    virtual void step(const FiniteAction& action, FiniteState& nextState, Reward& reward) override;

protected:
    arma::vec P;
    arma::vec R;
};

}

#endif /* INCLUDE_RELE_ENVIRONMENTS_MAB_SIMPLEMAB_H_ */
