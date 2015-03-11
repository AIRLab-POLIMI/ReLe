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

#ifndef INCLUDE_RELE_CORE_OPTIONS_H_
#define INCLUDE_RELE_CORE_OPTIONS_H_

#include "Basics.h"

namespace ReLe
{

enum OptionType
{
    Normal, Basic, Fixed
};

template<class ActionC, class StateC>
class AbstractOption
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

public:
    virtual bool canStart(const StateC& state) = 0;
    virtual double terminationProbability(const StateC& state) = 0;
    virtual void generateReward(const StateC& state, const ActionC& action, Reward& reward) = 0;

    virtual inline OptionType getOptionType()
    {
        return Normal;
    }

    virtual ~AbstractOption() {}

};

template<class ActionC, class StateC>
class BasicOption : public AbstractOption<ActionC, StateC>
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

public:
    virtual inline void generateReward(const StateC& state, const ActionC& action, Reward& reward) { }

    virtual inline OptionType getOptionType()
    {
        return Basic;
    }

    virtual ~BasicOption() {}

};

template<class ActionC, class StateC>
class FixedOption : public AbstractOption<ActionC, StateC>
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

public:
    virtual inline void generateReward(const StateC& state, const ActionC& action, Reward& reward) { }

    virtual inline OptionType getOptionType()
    {
        return Fixed;
    }

    virtual ~FixedOption() {}

};

}

#endif /* INCLUDE_RELE_CORE_OPTIONS_H_ */
