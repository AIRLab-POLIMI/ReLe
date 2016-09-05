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

#ifndef RANDOMPOLICY_H_
#define RANDOMPOLICY_H_

#include "rele/policy/Policy.h"
#include "rele/utils/Range.h"
#include "rele/utils/RandomGenerator.h"

#include <cassert>

namespace ReLe
{

/*!
 * This policy selects a purely random action given any input state.
 * Each action value is selected within the given range, i.e., it is
 * drawn from an uniform bounded distribution:
 * \f[ \pi(u|x) = \mathcal{U}(\Omega)\f]
 * where \f$\Omega \subseteq \mathbb{R}^{n}\f$.
 */
template<class StateC>
class RandomPolicy : public Policy<DenseAction,StateC>
{

protected:

    ///Vector of ranges, each one representing
    /// the range of the i-th element of an action,
    /// i.e., \f$u_i \sim U(Omega_i)\f$
    std::vector<Range> ranges;

public:
    /**
     * Construct a RandomPolicy object, initializing the action range.
     * \param ranges the range of the action elements
     */
    RandomPolicy(std::vector<Range>& ranges)
        : ranges(ranges)
    {

    }

    virtual ~RandomPolicy()
    {

    }

    // Policy interface
public:
    std::string getPolicyName() override
    {
        return std::string("RandomPolicy");
    }

    std::string printPolicy() override
    {
        return std::string("");
    }

    virtual arma::vec operator() (typename state_type<StateC>::const_type_ref state) override
    {
    	arma::vec u(ranges.size());

        for (int i = 0; i < ranges.size(); ++i)
        {
            double a = ranges[i].lo();
            double b = ranges[i].hi();
            u[i] = RandomGenerator::sampleUniform(a, b);
        }

        return u;
    }

    virtual double operator() (
        typename state_type<StateC>::const_type_ref state,
        const arma::vec& action) override
    {
        double prob = 1.0;
        std::vector<Range>::iterator it;
        for(it = ranges.begin(); it != ranges.end(); ++it)
        {
            prob *= 1.0 / it->width();
        }
        return prob;
    }

    virtual RandomPolicy<StateC>* clone() override
    {
        return new RandomPolicy<StateC>(*this);
    }

};

/*!
 * Random policy that produces action values according to a
 * discrete distribution, where each possible action has a predefined probability of being produced:
 * \f[\pi(u_i|x,D=[w_1,\dots,w_n]) = \frac{w_i}{\sum_{k=1}^{n} w_k}, \quad 1\leq i\leq n.\f]
 * The w's are a set of n non-negative individual weights set on construction (or using member param)
 * that defines the discrete action probability.
 */
template<class ActionC, class StateC>
class StochasticDiscretePolicy: public virtual Policy<ActionC,StateC>
{
protected:
    std::vector<ActionC> mActions;
    arma::vec distribution;
public:

    /*!
     * Constructs a StochasticDiscretePolicy object, initializing it with the given ActionList. The distribution
     * is set uniformly over the action contained in the action list. Note that the ActionList in cloned (deep-copy).
     * All weights shall be non-negative values, and at least one of the values in the sequence must be positive.
     *
     * \param actions a pointer to an action list
     * \param actionDim the dimension of each action
     */
    StochasticDiscretePolicy(std::vector<ActionC> actions) :
        mActions(actions), distribution(actions.size())
    {
        double nbel = mActions.size();
        double p = 1/nbel;
        for (int i = 0; i < nbel; ++i)
        {
            distribution[i] = p;
        }
    }

    /*!
     * Constructs a StochasticDiscretePolicy object, initializing it with the given ActionList and a set of weights
     * representing the distribution over actions.
     * All weights shall be non-negative values, and at least one of the values in the sequence must be positive.
     * They do not have to be 1-sum, normalization is performed in the construction phase.
     *
     * \param actions a pointer to an action list
     * \param dist an array storing the discrete distribution
     */
    StochasticDiscretePolicy(std::vector<ActionC> actions, double* dist) :
        mActions(actions), distribution(mActions.size())
    {
        double tot = 0.0;
        for (int i = 0; i < mActions.size(); ++i)
        {
            tot += dist[i];
            distribution[i] = dist[i];
        }
        for (int i = 0; i < mActions.size(); ++i)
        {
            distribution[i] /= tot;
            std::cout << distribution[i] << " ";
        }
        std::cout << std::endl;
    }

    virtual ~StochasticDiscretePolicy()
    {
        //        delete [] distribution;
    }

    std::string getPolicyName() override
    {
        return std::string("StochasticDiscretePolicy");
    }

    std::string printPolicy() override
    {
        return std::string("");
    }

    virtual double operator() (
        typename state_type<StateC>::const_type_ref state,
        typename action_type<ActionC>::const_type_ref action) override
    {
        int idx = findAction(action);
        return distribution[idx];
    }

    virtual typename action_type<ActionC>::type operator() (typename state_type<StateC>::const_type_ref state) override
    {
        std::size_t idx = RandomGenerator::sampleDiscrete(distribution.begin(), distribution.end());
        return mActions[idx];
    }

    virtual StochasticDiscretePolicy<ActionC, StateC>* clone() override
    {
        return new StochasticDiscretePolicy<ActionC, StateC>(*this);
    }

private:
    int findAction(typename action_type<ActionC>::const_type action)
    {
        typename action_type<ActionC>::type a1 = action;
        for (int i = 0, ie = mActions.size(); i < ie; ++i)
        {
            typename action_type<ActionC>::type a2 = mActions[i];
            if(a1 == a2)
            {
                return i;
            }
        }
        std::cerr << "Error: unknown action" << std::endl;
        std::cerr << "Action: " << action << std::endl;
        abort();
    }
};

} // end namespace

#endif // RANDOMPOLICY_H_
