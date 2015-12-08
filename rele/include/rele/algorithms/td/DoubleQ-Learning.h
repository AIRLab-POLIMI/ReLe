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

#include "Q-Learning.h"

namespace ReLe
{
/*
 * Double Q-Learning
 *
 * "Double Q-Learning"
 * Hado Van Hasselt
 */

class DoubleQ_Learning: public Q_Learning
{
public:
    DoubleQ_Learning(ActionValuePolicy<FiniteState>& policy);
    virtual void initEpisode(const FiniteState& state, FiniteAction& action) override;
    virtual void sampleAction(const FiniteState& state, FiniteAction& action) override;
    virtual void step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action) override;
    virtual void endEpisode(const Reward& reward) override;

    virtual ~DoubleQ_Learning();

protected:
    virtual void init() override;
    virtual void updateQ();

protected:
    arma::mat QA;
    arma::mat QB;

};

}

#endif /* INCLUDE_RELE_ALGORITHMS_TD_DOUBLEQ_LEARNING_H_ */
