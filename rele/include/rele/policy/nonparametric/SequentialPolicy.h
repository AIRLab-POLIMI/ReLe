/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta & Carlo D'Eramo
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

#include "rele/policy/q_policy/ActionValuePolicy.h"


namespace ReLe
{

/*!
 * This policy applies each action sequentially untill the end
 * of the episode. Then, it restarts from the first action.
 * It assumes that all the episodes have the same fixed length
 * which must be known at construction time.
 *
 * The length of the action sequence can be smaller than the
 * length of the episode. In this case, the action sequence is
 * reinitialized multiple times in an episode.
 * When the action sequence is longer than an episode, an action
 * sequence is executed over multiple episodes.
 */
class SequentialPolicy: public ActionValuePolicy<FiniteState>
{

public:
    /*!
     * Construct a Sequential policy with a fixed action sequence
     * and a fixed episode length.
     * \param nActions the number of actions in a sequence
     * \param episodeLength the length of the episodes
     */
    SequentialPolicy(unsigned int nActions, unsigned int episodeLength);
    unsigned int operator()(const size_t& state) override;
    double operator()(const size_t& state, const unsigned int& action) override;
    inline std::string getPolicyName() override;
    SequentialPolicy* clone() override;

protected:
    /// The action to be executed
    unsigned int currentAction;
    /// The currently observed step of an episode
    unsigned int currentEpisodeStep;
    /// The length of each episode (assumed fixed)
    unsigned int episodeLength;
};

}

#endif /* INCLUDE_RELE_POLICY_NONPARAMETRIC_SEQUENTIALPOLICY_H_ */
