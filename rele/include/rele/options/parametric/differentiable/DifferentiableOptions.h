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

#ifndef INCLUDE_RELE_OPTIONS_PARAMETRIC_DIFFERENTIABLE_DIFFERENTIABLEOPTIONS_H_
#define INCLUDE_RELE_OPTIONS_PARAMETRIC_DIFFERENTIABLE_DIFFERENTIABLEOPTIONS_H_


namespace ReLe
{


template<class ActionC, class StateC>
class DifferentiableOption : public Option<ActionC, StateC>
{
public:
    DifferentiableOption(DifferentiablePolicy<FiniteAction, StateC>& policy,
                         std::vector<Option<ActionC, StateC>*> options) : policy(policy)
    {

    }

    inline DifferentiablePolicy<FiniteAction, StateC>& getPolicy()
    {
        return policy;
    }

    virtual Option<ActionC, StateC>& operator ()(const StateC& state)
    {
        unsigned int index = policy(state);
        return *options[index];
    }

    virtual ~DifferentiableOption()
    {

    }

    virtual bool canStart(const StateC& state)
    {
        return true;
    }

    virtual double terminationProbability(const StateC& state)
    {
        return 0;
    }

protected:
    DifferentiablePolicy<FiniteAction, StateC>& policy;
    std::vector<Option<ActionC, StateC>*> options;

};

}

#endif /* INCLUDE_RELE_OPTIONS_PARAMETRIC_DIFFERENTIABLE_DIFFERENTIABLEOPTIONS_H_ */
