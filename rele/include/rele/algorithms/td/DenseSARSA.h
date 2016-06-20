/*
 * rele,
 *
 *
 * Copyright (C) 2015 Matteo Pirotta & Davide Tateo
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

#ifndef INCLUDE_ALGORITHMS_TD_DENSESARSA_H_
#define INCLUDE_ALGORITHMS_TD_DENSESARSA_H_

#include "rele/algorithms/td/TD.h"
#include <armadillo>

namespace ReLe
{

/*!
 * This class implements the linear SARSA algorithm.
 * This algorithm is an on-policy temporal difference algorithm.
 * Can only work on Dense MDP, i.e. with finite action and dense state space.
 *
 * References
 * ==========
 *
 * [Seijen, Sutton. True Online TD(lambda)](http://jmlr.org/proceedings/papers/v32/seijen14.pdf)
 */
class DenseSARSA: public LinearTD
{
    //TODO [IMPORTANT] change doc and fix implementation for non linear regressors

public:
    /*!
     * Constructor.
     * \param phi the features to be used for linear approximation of the state space
     * \param policy the policy to be used by the algorithm
     * \param alpha the learning rate to be used by the algorithm
     */
    DenseSARSA(Features& phi, ActionValuePolicy<DenseState>& policy, LearningRateDense& alpha);
    virtual void initEpisode(const DenseState& state, FiniteAction& action) override;
    virtual void sampleAction(const DenseState& state, FiniteAction& action) override;
    virtual void step(const Reward& reward, const DenseState& nextState,
                      FiniteAction& action) override;
    virtual void endEpisode(const Reward& reward) override;

    virtual ~DenseSARSA();

    inline void setLambda(double lambda)
    {
    	this->lambda = lambda;
    }

    inline void setReplacingTraces(bool val)
    {
        useReplacingTraces = val;
    }

private:
    arma::vec eligibility;
    double lambda;
    bool useReplacingTraces;

};

}

#endif /* INCLUDE_ALGORITHMS_TD_DENSESARSA_H_ */
