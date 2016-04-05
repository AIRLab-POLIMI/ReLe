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

#ifndef INCLUDE_RELE_ALGORITHMS_TD_DOUBLEQ_LEARNING_H_
#define INCLUDE_RELE_ALGORITHMS_TD_DOUBLEQ_LEARNING_H_

#include "rele/algorithms/td/Q-Learning.h"

namespace ReLe
{

/*!
 * This class implements the Double Q-Learning algorithm.
 * This algorithm is an off-policy temporal difference algorithm that
 * tries to solve the well-known overestimation problem suffered
 * by Q-Learning.
 * More precisely, this algorithm stores two Q-tables and uses one of them
 * to select the action that maximizes the Q-value, but uses the value of that
 * action stored in the other Q-Table. This technique has been proved to have
 * negative bias (as opposed to Q-Learning that has a positive bias) which can
 * help to avoid incremental approximation error caused by overestimations.
 * It can only work on finite MDP, i.e. with both finite action and state space.
 *
 * References
 * =========
 *
 * [Van Hasselt. Double Q-Learning](https://papers.nips.cc/paper/3964-double-q-learning.pdf)
 */
class DoubleQ_Learning: public Q_Learning
{
public:
    DoubleQ_Learning(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha);
    virtual void initEpisode(const FiniteState& state, FiniteAction& action) override;
    virtual void sampleAction(const FiniteState& state, FiniteAction& action) override;
    virtual void step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action) override;
    virtual void endEpisode(const Reward& reward) override;

    virtual ~DoubleQ_Learning();

protected:
    virtual void init() override;
    inline virtual void updateQ();

protected:
    arma::cube doubleQ;
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_TD_DOUBLEQ_LEARNING_H_ */
