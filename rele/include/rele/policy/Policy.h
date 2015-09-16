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

#ifndef POLICY_H_
#define POLICY_H_

#include "BasicsTraits.h"
#include "Interfaces.h"
#include "ActionMask.h"

#include <string>

namespace ReLe
{

template<class ActionC, class StateC>
class Policy
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

public:
    Policy()
    {
        actionMask = nullptr;
    }

    virtual typename action_type<ActionC>::type operator() (typename state_type<StateC>::const_type_ref state) = 0;
    virtual double operator() (typename state_type<StateC>::const_type_ref state, typename action_type<ActionC>::const_type_ref action) = 0;

    virtual std::string getPolicyName() = 0;
    virtual std::string getPolicyHyperparameters() = 0;
    virtual std::string printPolicy() = 0;

    virtual Policy<ActionC, StateC>* clone() = 0;

    void setMask(ActionMask<StateC, bool>* mask)
    {
        this->actionMask = mask;
    }

    virtual ~Policy()
    {

    }

protected:
    std::vector<bool> getMask(typename state_type<StateC>::const_type_ref state, unsigned int size)
    {
        if(actionMask)
            return actionMask->getMask(state);

        return std::vector<bool>(size, true);
    }

protected:
    ActionMask<StateC, bool>* actionMask; //FIXME add template parameter if needed, or enable if
};

template<class ActionC, class StateC>
class NonParametricPolicy: public Policy<ActionC, StateC>
{

};

template<class ActionC, class StateC>
class ParametricPolicy: public Policy<ActionC, StateC>
{
public:
    virtual arma::vec getParameters() const = 0;
    virtual const unsigned int getParametersSize() const = 0;
    virtual void setParameters(const arma::vec& w) = 0;

    virtual ~ParametricPolicy()
    {

    }

};

template<class ActionC, class StateC>
class DifferentiablePolicy: public ParametricPolicy<ActionC, StateC>
{
public:
    virtual arma::vec diff(typename state_type<StateC>::const_type_ref state, typename action_type<ActionC>::const_type_ref action) = 0;
    virtual arma::vec difflog(typename state_type<StateC>::const_type_ref state, typename action_type<ActionC>::const_type_ref action) = 0;

    virtual ~DifferentiablePolicy()
    {

    }
};

} //end namespace

#endif /* POLICY_H_ */
