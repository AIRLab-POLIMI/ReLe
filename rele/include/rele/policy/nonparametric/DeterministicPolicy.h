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

#include "Policy.h"

namespace ReLe
{

class DeterministicPolicy: public NonParametricPolicy<FiniteAction, FiniteState>
{

public:
    virtual unsigned int operator()(size_t state);
    virtual double operator()(size_t state, unsigned int action);

    inline virtual std::string getPolicyName()
    {
        return "Deterministic";
    }

    virtual std::string getPolicyHyperparameters()
    {
        return "";
    }

    virtual std::string printPolicy();

    inline void update(size_t state, unsigned int action)
    {
        pi[state] = action;
    }

    inline void init(size_t nstates)
    {
        pi.zeros(nstates);
    }

private:
    arma::uvec pi;
};


}


#endif /* INCLUDE_RELE_POLICY_NONPARAMETRIC_DETERMINISTICPOLICY_H_ */
