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


#ifndef INCLUDE_RELE_ALGORITHMS_TD_WQ_LEARNING_H_
#define INCLUDE_RELE_ALGORITHMS_TD_WQ_LEARNING_H_

#include "rele/algorithms/td/Q-Learning.h"
#include <boost/math/distributions/normal.hpp>


namespace ReLe
{

/*!
 * This class implements the Weighted Q-Learning algorithm.
 * This algorithm is an off-policy temporal difference algorithm that,
 * as Double Q-Learning, tries to solve the well-known overestimation problem suffered
 * by Q-Learning. This algorithm computes an estimate of the maximum action-value
 * approximating it as weighted sum of action-values where the weights are the probabilities
 * of the respective action-value to be the maximum.
 * While Q-Learning gives the best results when there is a single action-value that is
 * clearly the maximum and Double Q-Learning gives the best results when all action-values
 * have almost the same value, Weighted Q-Learning has been proved to work well in intermediate
 * cases giving, in general, better approximation than the other two approaches.
 *
 * References
 * =========
 *
 * ...
 */
class WQ_Learning: public Q_Learning
{
public:
    static constexpr double stdZeroValue = 1e-5;
    static constexpr double stdInfValue = 1e10;
    static constexpr double nTrapz = 100;
    static constexpr double sigmaBound = 5;

public:
    WQ_Learning(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha);
    virtual void initEpisode(const FiniteState& state, FiniteAction& action) override;
    virtual void sampleAction(const FiniteState& state, FiniteAction& action) override;
    virtual void step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action) override;
    virtual void endEpisode(const Reward& reward) override;

    virtual ~WQ_Learning();

protected:
    arma::mat idxs;
    arma::mat meanQ;
    arma::mat sampleStdQ;
    arma::mat weightsVar;
    arma::mat Q2;
    arma::mat nUpdates;

protected:
    virtual void init() override;
    inline void updateMeanAndSampleStdQ(double target);
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_TD_WQ_LEARNING_H_ */
