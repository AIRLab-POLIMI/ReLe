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

#ifndef INCLUDE_RELE_POLICY_NONPARAMETRIC_DETERMINISTICPOLICY_H_
#define INCLUDE_RELE_POLICY_NONPARAMETRIC_DETERMINISTICPOLICY_H_

#include "rele/policy/Policy.h"

namespace ReLe
{

/*!
 * This class represents a tabular deterministic policy. This means that for each
 * state a policy is deterministically chosen: \f[u = \pi(x)\f].
 * The policy is represented by a vector with dimension \f$|\mathcal{X}|\f$.
 */
class DeterministicPolicy: public NonParametricPolicy<FiniteAction, FiniteState>
{

public:
    virtual unsigned int operator()(const size_t& state) override;
    virtual double operator()(const size_t& state, const unsigned int& action) override;

    inline virtual std::string getPolicyName() override
    {
        return "Deterministic";
    }

    virtual std::string printPolicy() override;

    /*!
     * Associate an action to a specific state.
     * \param state the state
     * \param action the action to be associated to the state
     */
    inline void update(size_t state, unsigned int action)
    {
        pi[state] = action;
    }

    /*!
     * Initiate the policy represantation with the provided dimension.
     * Note that the dimension must equal the number of states.
     * \param nstates the number states
     */
    inline void init(size_t nstates)
    {
        pi.zeros(nstates);
    }

    virtual DeterministicPolicy* clone() override
    {
        return new DeterministicPolicy(*this);
    }

private:
    arma::uvec pi;
};


}


#endif /* INCLUDE_RELE_POLICY_NONPARAMETRIC_DETERMINISTICPOLICY_H_ */
