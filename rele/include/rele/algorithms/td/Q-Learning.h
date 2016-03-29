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

#ifndef INCLUDE_ALGORITHMS_TD_Q_LEARNING_H_
#define INCLUDE_ALGORITHMS_TD_Q_LEARNING_H_

#include "rele/algorithms/td/TD.h"

namespace ReLe
{

/*!
 * This class implements the tabular Q-learning algorithm.
 * This algorithm is an off-policy temporal difference algorithm.
 * Can only work on finite MDP, i.e. with both finite action and state space.
 *
 * References
 * ==========
 *
 * [Watkins, Dayan. Q-learning](http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)
 */
class Q_Learning: public FiniteTD
{
public:
    Q_Learning(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha);
    virtual void initEpisode(const FiniteState& state, FiniteAction& action) override;
    virtual void sampleAction(const FiniteState& state, FiniteAction& action) override;
    virtual void step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action) override;
    virtual void endEpisode(const Reward& reward) override;


    virtual ~Q_Learning();

};

}



#endif /* INCLUDE_ALGORITHMS_TD_Q_LEARNING_H_ */
