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

#include "Agent.h"
#include <armadillo>

namespace ReLe
{

class Q_Learning: public Agent<FiniteAction, FiniteState>
{
public:
    Q_Learning(std::size_t statesN, std::size_t actionN);
    virtual void initEpisode(const FiniteState& state, FiniteAction& action);
    virtual void sampleAction(const FiniteState& state, FiniteAction& action);
    virtual void step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action);
    virtual void endEpisode(const Reward& reward);
    virtual void endEpisode();

    void setAlpha(double alpha)
    {
        this->alpha = alpha;
    }

    void setEpsilon(double eps)
    {
        this->eps = eps;
    }

    virtual ~Q_Learning();

private:
    unsigned int policy(std::size_t x);
    void printStatistics();

private:
    //Action-value function
    arma::mat Q;

    //current an previous actions and states
    size_t x;
    unsigned int u;

    //algorithm parameters
    double alpha;
    double eps;

};

}



#endif /* INCLUDE_ALGORITHMS_TD_Q_LEARNING_H_ */
