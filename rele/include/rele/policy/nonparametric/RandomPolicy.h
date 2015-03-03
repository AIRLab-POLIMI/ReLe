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

#include "Policy.h"
#include "Range.h"
#include "RandomGenerator.h"

namespace ReLe
{

/**
 * This agent selects a purely random action given any input state.
 * Each action value is selected within the given range, i.e., it is
 * drawn from an uniform bounded distribution:
 * \f[ \pi(a|s) = \mathcal{U}(\Omega)\f]
 * where \f$\Omega \subseteq R^{n}\f$.
 * @brief Purely continuous random policy
 */
template<class StateC>
class RandomPolicy : public Policy<DenseAction,StateC>
{

protected:

    std::vector<Range> mpRanges;
    DenseAction mpAction;

public:
    /**
     * Construct a RandomPolicy object, initializing the action range.
     * @brief Construct RandomPolicy
     * @param ranges the range of the action elements
     */
    RandomPolicy(std::vector<Range>& ranges)
        : mpRanges(ranges), mpAction(ranges->size())
    {
        for (int i = 0; i < ranges->size(); ++i)
        {
            mpAction[i] = 0.0;
        }
    }

    virtual ~RandomPolicy()
    {
        delete [] mpAction;
    }

    // Policy interface
public:
    std::string getPolicyName()
    {
        return std::string(DETLINPOL_NAME);
    }

    std::string getPolicyHyperparameters()
    {
        return std::string("");
    }

    std::string printPolicy()
    {
        return std::string("");
    }

    virtual DenseAction operator() (const StateC state)
    {
        for (int i = 0; i < mpRanges->dimension(); ++i)
        {
            double a = mpRanges[i].min();
            double b = mpRanges[i].max();
            double val = a + (b - a) * RandomGenerator::sampleUniform(0,1);
            mpAction[i] =  val;
        }
    }

    virtual double operator() (
        typename state_type<StateC>::const_type state,
        typename action_type<DenseAction>::const_type action)
    {
        arma::vec output = this->operator ()(state);

        //TODO CONTROLLARE ASSEGNAMENTO
        DenseAction a;
        a.copy_vec(output);
        if (a.isAlmostEqual(action))
        {
            return 1.0;
        }
        return 0.0;
    }

    /**
     * @copydoc Policy::pi()
     */
    double pi(const double* state, const double* action)
    {
        double prob = 1.0;
        typedef typename RLLib::Ranges<T>::iterator rit;
        for(rit it = mpRanges->begin(); it != mpRanges->end(); ++it)
        {
            prob *= 1.0 / (*it)->length();
        }
        return prob;
    }

    /**
     * @copydoc Policy::sampleAction()
     */
    const double* sampleAction(const double* state)
    {
        for (int i = 0; i < mpRanges->dimension(); ++i)
        {
            T a = mpRanges->at(i)->min();
            T b = mpRanges->at(i)->max();
            T val = a + (b - a) * PGT::math::Random();
            mpAction[i] =  val;
        }
        return mpAction;
    }

};

/**
 * Random policy that produces action values according to a
 * discrete distribution, where each possible action has a predefined probability of being produced:
 * \f[\pi(a_i|s,D=[w_1,\dots,w_n]) = \frac{w_i}{\sum_{k=1}^{n} w_k}, \quad 1\leq i\leq n.\f]
 * The w's are a set of n non-negative individual weights set on construction (or using member param)
 * that defines the discrete action probability.
 * @brief Discrete random policy
 */
template<class T>
class StochasticDiscretePolicy: public virtual DiscreteActionPolicy
{
protected:
    ActionList* mActions;
    int mActionDim;
    double* distribution;
public:

    /**
     * Constructs a StochasticDiscretePolicy object, initializing it with the given ActionList. The distribution
     * is set uniformly over the action contained in the action list. Note that the ActionList in cloned (deep-copy).
     * All weights shall be non-negative values, and at least one of the values in the sequence must be positive.
     * @brief Construct a discrete policy
     * @param actions a pointer to an action list
     * @param actionDim the dimension of each action
     */
    StochasticDiscretePolicy(ActionList* actions, int actionDim) :
        mActions(new ActionList(actions->size())), mActionDim(actionDim), distribution(new double[actions->size()])
    {
        for (int i = 0; i < mActions->size(); ++i)
        {
            mActions->at(i) = new double[actionDim + 1];
            //            mActions->at(i) = (double*) realloc(actions->at(i), sizeof(double) * (actionDim + 1));
            memcpy(mActions->at(i), actions->at(i), sizeof(double)*actionDim);
            mActions->at(i)[actionDim] = i;
            distribution[i] = 1.0 / mActions->size();
            for (int g = 0; g < actionDim + 1; ++g)
            {
                std::cout << mActions->at(i)[g] << " ";
            }
            std::cout << std::endl;
        }
    }

    /**
     * Constructs a StochasticDiscretePolicy object, initializing it with the given ActionList and a set of weights
     * representing the distribution over actions.
     * All weights shall be non-negative values, and at least one of the values in the sequence must be positive.
     * They do not have to be 1-sum, normalization is performed in the construction phase.
     *
     * Note that the ActionList in cloned (deep-copy).
     * @brief Construct a discrete policy
     * @param actions a pointer to an action list
     * @param actionDim the dimension of each action
     * @param dist an array storing the discrete distribution
     */
    StochasticDiscretePolicy(ActionList* actions, int actionDim, double* dist) :
        mActions(new ActionList(actions->size())), mActionDim(actionDim), distribution(new double[mActions->size()])
    {
        T tot = 0.0;
        for (int i = 0; i < mActions->size(); ++i)
        {
            mActions->at(i) = new double[actionDim + 1];
            //            mActions->at(i) = (double*) realloc(actions->at(i), sizeof(double) * (actionDim + 1));
            memcpy(mActions->at(i), actions->at(i), sizeof(double)*actionDim);
            mActions->at(i)[actionDim] = i;
            tot += dist[i];
            distribution[i] = dist[i];
        }
        for (int i = 0; i < mActions->size(); ++i)
        {
            distribution[i] /= tot;
            std::cout << distribution[i] << " ";
        }
        std::cout << std::endl;
    }

    virtual ~StochasticDiscretePolicy()
    {
        mActions->Clear();
        delete mActions;
        delete [] distribution;
    }

    /**
     * @copydoc DiscreteActionPolicy::pi()
     */
    double pi(const double* state, const double* action)
    {
        int idx = action[mActionDim];
        return distribution[idx];
    }

    /**
     * @copydoc DiscreteActionPolicy::sampleAction()
     */
    const double* sampleAction(const double* state)
    {
        RLLib::Boundedness::checkDistribution(distribution, mActions->size());
        int idx = PGT::math::RandInt(mActions->size()+1);
        return mActions->at(idx);
        double random = PGT::math::Random();
        double sum = 0.0;
        for (typename ActionList::const_iterator a = mActions->begin(); a != mActions->end(); ++a)
        {
            int idx = (*a)[mActionDim];
            sum += distribution[idx];
            if (sum >= random)
                return (*a);
        }
        return mActions->at(mActions->size() - 1);
    }
};

//TODO
template<class T>
class RandomDiscreteBiasPolicy: public StochasticDiscretePolicy<T>
{
protected:
    const double* prev;
    typedef StochasticDiscretePolicy<T> Base;
public:
    RandomDiscreteBiasPolicy(ActionList* actions, int actionSize) :
        StochasticDiscretePolicy<T>(actions, actionSize), prev(actions->at(0))
    { }

    virtual ~RandomDiscreteBiasPolicy()
    { }

    /**
     * @copydoc StochasticDiscretePolicy::pi()
     */
    double pi(const double* action)
    {
        return Base::distribution->at(action[Base::mActionDim]);
    }

    /**
     * @copydoc StochasticDiscretePolicy::sampleAction()
     */
    const double* sampleAction(const double* state)
    {

        // 50% prev action
        if (Base::mActions->size() == 1)
            Base::distribution[0] = 1.0;
        else
        {
            for (typename ActionList::const_iterator a = Base::mActions->begin(); a != Base::mActions->end(); ++a)
            {
                int id = (*a)[Base::mActionDim];
                int previd = prev[Base::mActionDim];
                if (id == previd)
                    Base::distribution[id] = 0.5;
                else
                    Base::distribution[id] = 0.5 / (Base::mActions->size() - 1);
            }
        }
        // chose an action
        double random = PGT::math::Random();
        double sum = 0.0;
        for (typename ActionList::const_iterator a = Base::mActions->begin(); a != Base::mActions->end(); ++a)
        {
            int idx = (*a)[Base::mActionDim];
            sum += Base::distribution[idx];
            if (sum >= random)
            {
                prev = (*a);
                return prev;
            }
        }
        prev = Base::mActions->at(Base::mActions->size() - 1);

        return prev;
    }

};

} // end namespace

#endif // RANDOMPOLICY_H_
