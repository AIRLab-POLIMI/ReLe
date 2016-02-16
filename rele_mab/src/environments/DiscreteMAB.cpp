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

#include "rele_mab/environments/DiscreteMAB.h"


namespace ReLe
{

DiscreteMAB::DiscreteMAB(arma::vec P, arma::vec R, unsigned int horizon) :
    P(P),
    R(R),
    MAB(horizon)
{
    EnvironmentSettings& task = getWritableSettings();
    task.finiteActionDim = P.n_elem;
    task.continuosActionDim = 0;
}

DiscreteMAB::DiscreteMAB(arma::vec P, double r, unsigned int horizon) :
    P(P),
    MAB(horizon)
{
    R = arma::vec(P.n_elem, arma::fill::ones) * r;

    EnvironmentSettings& task = getWritableSettings();
    task.finiteActionDim = P.n_elem;
    task.continuosActionDim = 0;
}
/*
DiscreteMAB::DiscreteMAB(arma::vec P, unsigned int nArms, double minRange, double maxRange,
		unsigned int horizon) :
	P(P),
	MAB(horizon)
{
	R = arma::vec(nArms);
	for (unsigned int i = 0; i < nArms; i++)
	{
		R(i) = minRange + (maxRange - minRange) * (1.0 * i) / (nArms - 1);
	}

	EnvironmentSettings& task = getWritableSettings();
    task.finiteActionDim = P.n_elem;
    task.continuosActionDim = 0;
}
*/
DiscreteMAB::DiscreteMAB(unsigned int nArms, double r, unsigned int horizon) :
    MAB(horizon)
{
    P = arma::vec(nArms, arma::fill::randu);
    R = arma::vec(P.n_elem, arma::fill::ones) * r;

    EnvironmentSettings& task = getWritableSettings();
    task.finiteActionDim = nArms;
    task.continuosActionDim = 0;
}

DiscreteMAB::DiscreteMAB(unsigned int nArms, unsigned int horizon) :
    MAB(horizon)
{
    P = arma::vec(nArms, arma::fill::randu);
    R = arma::vec(nArms, arma::fill::randn);

    EnvironmentSettings& task = getWritableSettings();
    task.finiteActionDim = nArms;
    task.continuosActionDim = 0;
}

arma::vec DiscreteMAB::getP()
{
    return P;
}


void DiscreteMAB::step(const FiniteAction& action, FiniteState& nextState, Reward& reward)
{
    nextState.setStateN(0);

    if(RandomGenerator::sampleEvent(P(action)))
        reward[0] = R(action);
    else
        reward[0] = 0;
}

}
