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

#ifndef INCLUDE_RELE_CORE_BASICFUNCTIONS_H_
#define INCLUDE_RELE_CORE_BASICFUNCTIONS_H_

#include "Basics.h"
#include "BasicsTraits.h"
#include "Policy.h"

namespace ReLe
{

//templates needed to store action and state in a vector
template<class StateC, class ActionC>
arma::vec vectorize(const StateC& state, const ActionC& action)
{
    return arma::join_vert(state, action);
}

template<class ActionC>
arma::vec vectorize(const FiniteState& state, const ActionC& action)
{
    arma::vec aux(1);
    aux[0] = state.getStateN();
    return arma::join_vert(aux, action);
}

template<class StateC>
arma::vec vectorize(const StateC& state, const FiniteAction& action)
{
    arma::vec aux(1);
    aux[0] = action.getActionN();
    return arma::join_vert(state, aux);
}

inline arma::vec vectorize(const FiniteState& state, const FiniteAction& action)
{
    arma::vec vec(2);
    vec[0] = state.getStateN();
    vec[1] = action.getActionN();
    return vec;
}

//templates needed to store action, state and next state in a vector
template<class StateC, class ActionC>
arma::vec vectorize(const StateC& state, const ActionC& action, const StateC& nextState)
{
    return arma::join_vert(arma::join_vert(state, action), nextState);
}

template<class ActionC>
arma::vec vectorize(const FiniteState& state, const ActionC& action, const FiniteState& nextState)
{
    arma::vec aux1(1);
    aux1[0] = state.getStateN();
    arma::vec aux2(1);
    aux2[0] = state.getStateN();
    return arma::join_vert(arma::join_vert(aux1, action), aux2);
}

template<class StateC>
arma::vec vectorize(const StateC& state, const FiniteAction& action, const StateC& nextState)
{
    arma::vec aux(1);
    aux[0] = action.getActionN();
    return arma::join_vert(arma::join_vert(state, aux), nextState);
}

inline arma::vec vectorize(const FiniteState& state, const FiniteAction& action, const FiniteState& nextState)
{
    arma::vec vec(3);
    vec[0] = state.getStateN();
    vec[1] = action.getActionN();
    vec[2] = state.getStateN();
    return vec;
}

}



#endif /* INCLUDE_RELE_CORE_BASICFUNCTIONS_H_ */
