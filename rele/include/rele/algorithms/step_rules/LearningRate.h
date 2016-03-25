/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#ifndef INCLUDE_RELE_ALGORITHMS_STEP_RULES_LEARNINGRATE_H_
#define INCLUDE_RELE_ALGORITHMS_STEP_RULES_LEARNINGRATE_H_

#include "rele/core/Basics.h"

#include <sstream>

namespace ReLe
{

/*!
 * This class implement a learning rate, with eventually decay rules
 * The learning rate can be action and state dependent
 */
template<class ActionC, class StateC>
class LearningRate_
{
public:
    /*!
     * Computes the learning rate
     */
    virtual double operator()(StateC x, ActionC u) = 0;

    /*!
     * Resets the learning rate to it's original value
     */
    virtual void reset() = 0;

    /*!
     * Writes the learning rate as a string
     */
    virtual std::string print() = 0;

    /*!
     * Destructor.
     */
    virtual ~LearningRate_()
    {

    }
};

//! Template alias.
typedef LearningRate_<FiniteAction, FiniteState> LearningRate;

//! Template alias.
typedef LearningRate_<FiniteAction, DenseState> LearningRateDense;


/*!
 * Implementation of a constant learning rate
 */
template<class ActionC, class StateC>
class ConstantLearningRate_ : public LearningRate_<ActionC, StateC>
{
public:
    /*!
     * Constructor.
     * \param alpha the value for the learning rate
     */
    ConstantLearningRate_(double alpha) : alpha(alpha)
    {

    }

    virtual double operator()(StateC x, ActionC u) override
    {
        return alpha;
    }

    virtual void reset() override
    {

    }

    virtual std::string print() override
    {
        return std::to_string(alpha);
    }

    virtual ~ConstantLearningRate_()
    {

    }

private:
    double alpha;

};

//! Template alias.
typedef ConstantLearningRate_<FiniteAction, FiniteState> ConstantLearningRate;

//! Template alias.
typedef ConstantLearningRate_<FiniteAction, DenseState> ConstantLearningRateDense;


/*!
 * Implementation of a simple decaying learning rate.
 * The learning rate follows the rule:
 * \f[ \alpha(t)=\min\left(\alpha_{min},\dfrac{\alpha(t_{0})}{t^{\omega}}\right) \f]
 *
 */
template<class ActionC, class StateC>
class DecayingLearningRate_ : public LearningRate_<ActionC, StateC>
{
public:
    /*!
     * Constructor.
     * \param initialAlpha the initial value for the learning rate
     * \param omega the exponent used to weight the time decay. \f$\omega\in\left(0,1\right]\f$
     */
    DecayingLearningRate_(double initialAlpha, double omega = 1.0, double minAlpha = 0.0)
        : initialAlpha(initialAlpha), omega(omega), minAlpha(minAlpha)
    {
        t = 0;
    }

    virtual double operator()(StateC x, ActionC u) override
    {
        t++;
        double alpha = initialAlpha/std::pow(static_cast<double>(t), omega);
        return std::max(minAlpha, alpha);
    }

    virtual void reset() override
    {
        t = 0;
    }

    virtual std::string print() override
    {
        std::stringstream ss;
        ss << "Decaying rate -- t:" << t << " alpha0: " << initialAlpha
           << " alphaInf: " << minAlpha << " omega: " << omega;
        return ss.str();
    }

    virtual ~DecayingLearningRate_()
    {

    }

private:
    double initialAlpha;
    double minAlpha;
    double omega;

    unsigned int t;

};

//! Template alias.
typedef DecayingLearningRate_<FiniteAction, FiniteState> DecayingLearningRate;

//! Template alias.
typedef DecayingLearningRate_<FiniteAction, DenseState> DecayingLearningRateDense;


}

#endif /* INCLUDE_RELE_ALGORITHMS_STEP_RULES_LEARNINGRATE_H_ */
