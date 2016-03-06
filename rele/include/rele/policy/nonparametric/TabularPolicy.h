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

#ifndef INCLUDE_RELE_POLICY_NONPARAMETRIC_TABULARPOLICY_H_
#define INCLUDE_RELE_POLICY_NONPARAMETRIC_TABULARPOLICY_H_

#include "rele/policy/Policy.h"

namespace ReLe
{

/*!
 * This class implements a tabular policy, i.e., a finete state and action policy.
 * The policy is represented as a probabilistic matrix \f[\pi \in \mathcal{R}^{\mathcal{X} \times \mathcal{U}}\f],
 * where each entry \f$\pi(x,u) \in [0,1]\f$ is the probability to execute action \f$u\f$ in
 * state \f$x\f$.
 */
class TabularPolicy: public NonParametricPolicy<FiniteAction, FiniteState>
{

public:
    class updater;

public:
    virtual unsigned int operator()(const size_t& state) override;
    virtual double operator()(const size_t& state, const unsigned int& action) override;

    inline virtual std::string getPolicyName() override
    {
        return "Tabular";
    }

    virtual std::string printPolicy() override;

    /*!
     * Update the action distribution for the provided state
     * \param state the state to be updated.
     * \return an updater object
     */
    updater update(size_t state)
    {
        return updater(pi.row(state));
    }

    /*!
     * Initialize the probability matrix with the provided dimensions.
     * For each state a uniform probability over the action is defined.
     * \param nstates the number of states (i.e., rows)
     * \param nactions the number of action (i.e., columns)
     */
    inline void init(size_t nstates, unsigned int nactions)
    {
        pi.set_size(nstates, nactions);
        pi.fill(1.0/nactions);
    }

    virtual TabularPolicy* clone() override
    {
        return new TabularPolicy(*this);
    }


public:
    /*!
     * Utility class used to update the tabular distribution of
     * a specific state. It maintains a vector that represents
     * the tabular distribution. An index, named currentIndex, is maintained
     * in order to allow an iterative procedure for the update.
     *
     * \see{update.operator <<();}
     */
    class updater
    {
        /*!
         * Initiate the internal distribution from the provided
         * vector. The currentIndex is set to the first element.
         * \param row the tabular distribution
         */
        updater(arma::subview_row<double>&& row);

    public:
        friend updater TabularPolicy::update(size_t state);

        /*!
         * Update the probability of the current action:
         * \f$\pi(currentIndex) = weight\f$.
         * After the update the currentIndex is incremented in
         * order to point to the subsequent element of the distribution.
         * \param weight the probability/frequency to be set
         */
        void operator<<(double weight);

        /*!
         * Check if all the distrubution has been updated.
         * \return True if there are elements to be updated, False otherwise.
         */
        inline bool toFill()
        {
            return currentIndex != nactions;
        }

        /*!
         * Return the current indix of the tabular
         * distribution over actions in a state.
         * \return the current position of the distribution
         * \see {updater.operator <<();}
         */
        inline unsigned int getCurrentState()
        {
            return currentIndex;
        }

        /*!
         * Normalize the distribution so that it
         * sums to one.
         */
        void normalize();

    private:
        arma::subview_row<double> row;
        unsigned int nactions;
        ///Index of the action probability to be processed
        unsigned int currentIndex;
    };

private:
    arma::mat pi;
};

}

#endif /* INCLUDE_RELE_POLICY_NONPARAMETRIC_TABULARPOLICY_H_ */
