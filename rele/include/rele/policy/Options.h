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

#ifndef INCLUDE_RELE_CORE_OPTIONS_H_
#define INCLUDE_RELE_CORE_OPTIONS_H_

#include "rele/core/Basics.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/core/BasicsTraits.h"

namespace ReLe
{

enum OptionType
{
    Normal, Basic, Fixed
};

template<class ActionC, class StateC>
class Option
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

public:
    virtual Option<ActionC, StateC>& operator ()(const StateC& state) = 0;
    virtual bool canStart(typename state_type<StateC>::const_type_ref state) = 0;
    virtual double terminationProbability(const StateC& state) = 0;

    virtual FiniteAction getLastChoice() //TODO support continuos options?
    {
        return lastChoice;
    }

    virtual void operator ()(const StateC& state, ActionC& action)
    {
        Option<ActionC, StateC>& self = *this;
        Option<ActionC, StateC>& child = self(state);
        child(state, action);
    }

    virtual void accumulateReward(const Reward& newReward, double dF)
    {
        if(reward.size() != newReward.size())
        {
            reward.resize(newReward.size(), 0);
        }

        for(int i = 0; i < reward.size(); i++)
        {
            reward[i] += dF*newReward[i];
        }
    }

    virtual void getReward(Reward& r)
    {
        r = reward;
        reward.clear();
    }

    virtual bool hasEnded(const StateC& state)
    {
        double pT = terminationProbability(state);
        return state.isAbsorbing() || RandomGenerator::sampleEvent(pT);
    }

    virtual inline OptionType getOptionType()
    {
        return Normal;
    }

    virtual ~Option() {}

protected:
    Reward reward;
    FiniteAction lastChoice;

};

template<class ActionC, class StateC>
class BasicOption : public Option<ActionC, StateC>
{

public:
    virtual Option<ActionC, StateC>& operator ()(const StateC& state)
    {
        throw std::logic_error("Basic options cannot generate sub options");
    }

    virtual inline OptionType getOptionType()
    {
        return Basic;
    }

    virtual ~BasicOption() {}

};

template<class ActionC, class StateC>
class FixedOption : public Option<ActionC, StateC>
{

public:
    virtual Option<ActionC, StateC>& operator ()(const StateC& state) override
    {
        throw std::logic_error("Fixed options cannot generate sub options");
    }

    virtual inline OptionType getOptionType() override
    {
        return Fixed;
    }

    virtual ~FixedOption() {}

};

template<class ActionC, class StateC> using OptionStack = std::vector<Option<ActionC, StateC>*>;

}

#endif /* INCLUDE_RELE_CORE_OPTIONS_H_ */
