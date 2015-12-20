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

#ifndef INCLUDE_RELE_POLICY_NONPARAMETRIC_SEQUENTIALPOLICY_H_
#define INCLUDE_RELE_POLICY_NONPARAMETRIC_SEQUENTIALPOLICY_H_

#include "q_policy/ActionValuePolicy.h"


namespace ReLe
{

class SequentialPolicy: public ActionValuePolicy<FiniteState>
{

    /*
     * This policy applies each action sequentially till the end
     * of the episode. Then, it restarts from the first action.
     */

public:
    SequentialPolicy(unsigned int nActions, unsigned int episodeLength);
    unsigned int operator()(const size_t& state) override;
    double operator()(const size_t& state, const unsigned int& action) override;
    inline std::string getPolicyName() override;
    std::string getPolicyHyperparameters() override;
    SequentialPolicy* clone() override;

protected:
    unsigned int currentAction;
    unsigned int currentEpisodeStep;
    unsigned int episodeLength;
};

}

#endif /* INCLUDE_RELE_POLICY_NONPARAMETRIC_SEQUENTIALPOLICY_H_ */
