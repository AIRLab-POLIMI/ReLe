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

#ifndef LQR_H_
#define LQR_H_

#include "rele/core/ContinuousMDP.h"

namespace ReLe
{

/*!
 * This class implements a Linear-Quadratic Regulator.
 * This task aims to minimize the undesired deviations
 * from nominal values of some controller settings
 * in control problems.
 *
 * References
 * ==========
 * [Parisi, Pirotta, Smacchia, Bascetta, Restelli. Policy gradient approaches for multi-objective sequential decision making. IJCNN 2014](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6889738)
 */
class LQR: public ContinuousMDP
{
    friend class LQRsolver;
    friend class LQRExact;
public:
    enum S0Type {FIXED, RANDOM};

    /*!
     * Constructor.
     * \param dimension MDP dimension
     * \param reward_dimension reward dimension
     * \param eps
     * \param gamma MDP discount factor
     * \param horizon MDP horizon
     */
    LQR(unsigned int dimension, unsigned int reward_dimension,
        double eps = 0.1, double gamma = 0.9, unsigned int horizon = 50);

    /*!
     * Constructor.
     * \param A initialization matrix
     * \param B initialization matrix
     * \param Q
     * \param R reward matrix
     * \param gamma MDP discount factor
     * \param horizon MDP horizon
     */
    LQR(arma::mat& A, arma::mat& B, std::vector<arma::mat>& Q, std::vector<arma::mat>& R,
        double gamma = 0.9, unsigned int horizon = 50);

    /*!
     * \see Environment::step
     */
    virtual void step(const DenseAction& action, DenseState& nextState,
                      Reward& reward) override;

    /*!
     * \see Environment::getInitialState
     */
    virtual void getInitialState(DenseState& state) override;

public:
    /*!
     * Setter.
     * Set the initial state
     * \param initialState initial state
     */
    void setInitialState(arma::vec& initialState)
    {
        this->initialState = initialState;
    }

private:
    void initialize(unsigned int stateActionSize, unsigned int rewardSize, double e);
    void setInitialState();

private:
    arma::mat A, B;
    std::vector<arma::mat> Q, R;
    arma::vec initialState;
    S0Type startType;
};

}

#endif /* LQR_H_ */
