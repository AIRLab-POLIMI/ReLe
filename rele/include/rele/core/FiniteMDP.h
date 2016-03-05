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

#ifndef FINITEMDP_H_
#define FINITEMDP_H_

#include "rele/core/Environment.h"
#include "rele/core/Basics.h"

#include <set>
#include <armadillo>

namespace ReLe
{

/*!
 * This class implements a finite MDP, that is an MDP with both finite actions and states.
 * A finite MDP can be described by the tuple
 * \f$MDP=\left\langle \mathcal{S},\mathcal{A},\mathcal{P}_{a}(.,..),\mathcal{R}_{a}(.,.),\gamma\right\rangle$\f
 * where \f$\mathcal{S}$\f is the set of states, \f$\mathcal{A}$\f is the set of actions,
 * \f$\mathcal{P}_{a}:\mathcal{A}\times\mathcal{S}\times\mathcal{S}\rightarrow\mathbb{R}$\f is the transition function,
 * \f$\mathcal{R}_{a}:\mathcal{A}\times\mathcal{S}\times\mathcal{S}\rightarrow\mathbb{R}$\f is the reward function.
 * This class also support gaussian reward functions, so it's possible to specify the variance of the reward.
 * A finite MDP can also be solved exactly (or approximately) by a dynamic programming solver.
 */
class FiniteMDP: public Environment<FiniteAction, FiniteState>
{
    friend class DynamicProgrammingAlgorithm;
public:
    /*!
     * Constructor
     * \param P the transition function, a \f$\mathcal{A}\times\mathcal{S}\times\mathcal{S}$\f cube
     * \param R the reward function a \f$\mathcal{A}\times\mathcal{S}\times\mathcal{S}$\f cube
     * \param Rsigma the reward function variance in each state a \f$\mathcal{A}\times\mathcal{S}\times\mathcal{S}$\f cube
     * \param isFiniteHorizon if the mdp has a finite horizon
     * \param gamma the discount factor
     * \param horizon the mdp horizon
     */
    FiniteMDP(arma::cube P, arma::cube R, arma::cube Rsigma,
              bool isFiniteHorizon, double gamma = 1.0, unsigned int horizon = 0);

    /*!
     * Constructor
     * \param P the transition function, a \f$\mathcal{A}\times\mathcal{S}\times\mathcal{S}$\f cube
     * \param R the reward function a \f$\mathcal{A}\times\mathcal{S}\times\mathcal{S}$\f cube
     * \param Rsigma the reward function variance in each state a \f$\mathcal{A}\times\mathcal{S}\times\mathcal{S}$\f cube
     * \param settings the environment settings
     */
    FiniteMDP(arma::cube P, arma::cube R, arma::cube Rsigma, EnvironmentSettings* settings);

    /*!
     * \see Agent::step
     */
    virtual void step(const FiniteAction& action, FiniteState& nextState,
                      Reward& reward) override;

    /*!
     * \see Agent::getInitialState
     */
    virtual void getInitialState(FiniteState& state) override;

private:
    void chekMatricesDimensions(const arma::cube& P, const arma::cube& R,
                                const arma::cube& Rsigma);
    void setupenvironment(bool isFiniteHorizon, unsigned int horizon,
                          double gamma, const arma::cube& P);
    void findAbsorbingStates();

private:
    arma::cube P;
    arma::cube R;
    arma::cube Rsigma;
    FiniteState currentState;
    std::set<unsigned int> absorbingStates;

};

}

#endif /* FINITEMDP_H_ */
