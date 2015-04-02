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

#ifndef INCLUDE_RELE_CORE_BASICFUNCTIONS_H_
#define INCLUDE_RELE_CORE_BASICFUNCTIONS_H_

#include "Basics.h"
#include "BasicsTraits.h"

namespace ReLe
{
//Templates needed to sample different action types
template<class PolicyC>
void sampleActionWorker(const FiniteState& state, FiniteAction& action, PolicyC& policy)
{
    unsigned int u = policy(state.getStateN());
    action.setActionN(u);
}


template<class StateC, class PolicyC>
void sampleActionWorker(const StateC& state, FiniteAction& action, PolicyC& policy)
{
    unsigned int u = policy(state);
    action.setActionN(u);
}

template<class StateC, class ActionC, class PolicyC>
void sampleActionWorker(const StateC& state, ActionC& action, PolicyC& policy)
{
    typename action_type<ActionC>::type_ref u = action;
    u = policy(state);
}

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

}



#endif /* INCLUDE_RELE_CORE_BASICFUNCTIONS_H_ */
