/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef INCLUDE_RELE_POLICY_PARAMETRIC_DIFFERENTIABLE_NEWGIBBSPOLICY_H_
#define INCLUDE_RELE_POLICY_PARAMETRIC_DIFFERENTIABLE_NEWGIBBSPOLICY_H_


#include "Policy.h"
#include "regressors/LinearApproximator.h"
#include "RandomGenerator.h"

#include <stdexcept>

namespace ReLe
{

template<class StateC>
class NewGibbsPolicy : public DifferentiablePolicy<FiniteAction, StateC>
{
public:

    NewGibbsPolicy(std::vector<FiniteAction> actions,
                   Features& phi, double temperature) :
        mActions(actions), distribution(actions.size(),0),
        approximator(phi), tau(temperature)
    {
    }

    virtual ~NewGibbsPolicy()
    {
    }

    inline void setTemperature(double temperature)
    {
        tau = temperature;
    }


    // Policy interface
public:
    std::string getPolicyName()
    {
        return std::string("NewGibbsPolicy");
    }

    std::string getPolicyHyperparameters()
    {
        return std::string("");
    }

    std::string printPolicy()
    {
        return std::string("");
    }


    double operator() (typename state_type<StateC>::const_type_ref state,
                       const unsigned int& action)
    {
        int statesize = state.size();
        int nactions = mActions.size();

        arma::vec tuple(1+statesize);
        tuple(arma::span(0, statesize-1)) = state;

        arma::vec&& distribution = computeDistribution(nactions, tuple,	statesize);
        unsigned int index = findActionIndex(action);
        return distribution[index];
    }

    unsigned int operator() (typename state_type<StateC>::const_type_ref state)
    {
        int statesize = state.size();
        int nactions = mActions.size();

        arma::vec tuple(1+statesize);
        tuple(arma::span(0, statesize-1)) = state;


        arma::vec&& distribution = computeDistribution(nactions, tuple,	statesize);
        unsigned int idx = RandomGenerator::sampleDiscrete(distribution.begin(), distribution.end());
        return mActions.at(idx).getActionN();
    }

    virtual NewGibbsPolicy<StateC>* clone()
    {
        return new  NewGibbsPolicy<StateC>(*this);
    }

    // ParametricPolicy interface
public:
    virtual inline arma::vec getParameters() const
    {
        return approximator.getParameters();
    }

    virtual inline const unsigned int getParametersSize() const
    {
        return approximator.getParametersSize();
    }

    virtual inline void setParameters(arma::vec& w)
    {
        approximator.setParameters(w);
    }

    // DifferentiablePolicy interface
public:
    virtual arma::vec diff(typename state_type<StateC>::const_type_ref state,
                           typename action_type<FiniteAction>::const_type_ref action)
    {
        NewGibbsPolicy& pi = *this;
        return pi(state, action)*difflog(state, action);
    }

    virtual arma::vec difflog(typename state_type<StateC>::const_type_ref state,
                              typename action_type<FiniteAction>::const_type_ref action)
    {
        int statesize = state.size();
        int nactions = mActions.size();

        arma::vec tuple(1+statesize);
        tuple(arma::span(0, statesize-1)) = state;


        arma::vec&& distribution = computeDistribution(nactions, tuple, statesize);
        Features& phi = approximator.getBasis();
        arma::mat features(phi.rows(), nactions);

        for (unsigned int k = 0; k < nactions; k++)
        {
            tuple[statesize] = mActions[k].getActionN();
            features.col(k) = phi(tuple)/ tau;
        }

        unsigned int index = findActionIndex(action);

        return features.col(index) - features*distribution;
    }


private:
    arma::vec computeDistribution(int nactions, arma::vec tuple, int statesize)
    {
        double den = 0.0;
        arma::vec distribution(nactions);
        for (unsigned int k = 0; k < nactions; k++)
        {
            tuple[statesize] = mActions[k].getActionN();
            arma::vec preference = approximator(tuple);
            double val = exp(preference[0] / tau);
            den += val;
            distribution[k] = val;
        }
        distribution /= den;
        return distribution;
    }

    unsigned int findActionIndex(const unsigned int& action)
    {
        unsigned int index;

        for (index = 0; index < mActions.size(); index++)
        {
            if (action == mActions[index].getActionN())
                break;
        }

        if (index == mActions.size())
        {
            throw std::runtime_error("Action not found");
        }

        return index;
    }

private:
    std::vector<FiniteAction> mActions;
    std::vector<double> distribution;
    double tau;
    LinearApproximator approximator;
};


}

#endif /* INCLUDE_RELE_POLICY_PARAMETRIC_DIFFERENTIABLE_NEWGIBBSPOLICY_H_ */
