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


#include "parametric/differentiable/GenericNormalPolicy.h"
#include "RandomGenerator.h"

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// MVN POLICY
///////////////////////////////////////////////////////////////////////////////////////

double GenericMVNPolicy::operator()(const arma::vec& state, const arma::vec& action)
{
    updateInternalState(state);
    return mvnpdfFast(action, mean, invSigma, determinant);
}

arma::vec GenericMVNPolicy::operator() (const arma::vec& state)
{
    updateInternalState(state);
    return mvnrandFast(mean, choleskySigma);
}

arma::vec GenericMVNPolicy::diff(const arma::vec &state, const arma::vec &action)
{
    return (*this)(state,action) * difflog(state,action);
}

arma::vec GenericMVNPolicy::difflog(const arma::vec &state, const arma::vec &action)
{
    updateInternalState(state);

    arma::vec delta = action - mean;
    arma::mat gMu = approximator.diff(state);
    gMu.reshape(approximator.getParametersSize(), mean.n_elem);

    // compute gradient
    return 0.5 * gMu * (invSigma + invSigma.t()) * delta;
}

arma::mat GenericMVNPolicy::diff2log(const arma::vec& state, const arma::vec& action)
{
	//TODO IMPLEMENT!
	return arma::mat();
}

}
