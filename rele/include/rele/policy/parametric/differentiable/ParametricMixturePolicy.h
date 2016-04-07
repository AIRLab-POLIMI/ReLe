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

#ifndef PARAMETRICMIXTUREPOLICY_H_
#define PARAMETRICMIXTUREPOLICY_H_

#include "rele/policy/Policy.h"
#include "rele/utils/RandomGenerator.h"
#include <cassert>

namespace ReLe
{

/**
 * \f[ M(x) = \sum_{i=1}^N \alpha_i \pi_i(x) \f]
 * but \f$ \sum_{i=1}^{N} \alpha_i = 1 \f$
 * then
 * \f[ M(x) = \sum_{i=1}^{N-1} \alpha_i \pi_i(x) + (1-\sum_{i=1}^{N-1} \alpha_i) \pi_{N}(x) \f]
 */
template<class ActionC, class StateC>
class GenericParametricMixturePolicy : public DifferentiablePolicy<ActionC, StateC>
{
public:
    GenericParametricMixturePolicy(std::vector<DifferentiablePolicy<ActionC, StateC>*>& mixture)
        : mixture(mixture)
    {
        int n = mixture.size();
        // note that the number of coefficients is N-1 due to 1-sum constraint
        // initial mixture is uniform
        mixtureWeights = arma::ones(n-1) / n;
    }

    GenericParametricMixturePolicy(std::vector<DifferentiablePolicy<ActionC, StateC>*>& mixture, arma::vec coeff)
        : mixture(mixture), mixtureWeights(arma::abs(coeff.rows(0,mixture.size()-2)))
    {
        assert(mixtureWeights.n_elem >= mixture.size() - 1);
        mixtureWeights /= arma::sum(mixtureWeights);
    }

    virtual ~GenericParametricMixturePolicy()
    {

    }

    // Policy interface
public:
    virtual typename action_type<ActionC>::type operator() (typename state_type<StateC>::const_type_ref state) override
    {
        arma::vec preferences = computeParameters(mixtureWeights);
        std::size_t idx = RandomGenerator::sampleDiscrete<arma::vec::iterator>(preferences.begin(), preferences.end());
        return mixture[idx]->operator()(state);
    }

    virtual double operator() (typename state_type<StateC>::const_type_ref state, typename action_type<ActionC>::const_type_ref action) override
    {
        arma::vec preferences = computeParameters(mixtureWeights);

        unsigned int nbElem = mixture.size();
        double probability = 0.0;
        for (int i = 0; i < nbElem; ++i)
        {
            probability += preferences(i)*mixture[i]->operator()(state,action);
        }
        return probability;
    }

    virtual std::string getPolicyName() override
    {
        return "GenericParametricMixturePolicy";
    }

    virtual std::string printPolicy() override
    {
        return "";
    }

    virtual Policy<ActionC,StateC>* clone() override
    {
        return new GenericParametricMixturePolicy(this->mixture, this->mixtureWeights);
    }

    // ParametricPolicy interface
public:
    virtual arma::vec getParameters() const override
    {
        unsigned int nbElem = mixture.size();
        arma::vec parameters;
        for (int i = 0; i < nbElem; ++i)
        {
            parameters = arma::join_vert(parameters, mixture[i]->getParameters());
        }
        parameters = arma::join_vert(parameters, mixtureWeights);
        return parameters;
    }

    virtual const unsigned int getParametersSize() const override
    {
        unsigned int nbElem = mixture.size();
        unsigned int nbParams = mixtureWeights.n_elem;
        for (int i = 0; i < nbElem; ++i)
        {
            nbParams += mixture[i]->getParametersSize();
        }
        return nbParams;
    }

    virtual void setParameters(const arma::vec& w) override
    {
        unsigned int nbElem = mixture.size();
        unsigned int idx = 0;
        for (int i = 0; i < nbElem; ++i)
        {
            mixture[i]->setParameters(w.rows(idx,  idx+mixture[i]->getParametersSize()-1));
            idx +=  mixture[i]->getParametersSize();
        }
        mixtureWeights = w.rows(idx, idx+mixtureWeights.n_elem-1);
    }

    // DifferentiablePolicy interface
public:
    virtual arma::vec difflog(typename state_type<StateC>::const_type_ref state, typename action_type<ActionC>::const_type_ref action) override
    {
        int nbParams = mixtureWeights.n_elem;
        arma::vec preferences = computeParameters(mixtureWeights);
        arma::vec lgMix, lgCoeff(nbParams);
        unsigned int nbElem = mixture.size();
        for (int i = 0; i < nbElem; ++i)
        {
            arma::vec gP = preferences(i) * mixture[i]->diff(state,action);
            lgMix = arma::join_vert(lgMix, gP);
        }
        double pval = mixture[nbElem-1]->operator()(state,action);
        for (int i = 0; i < nbParams; ++i)
        {
            lgCoeff(i) = mixture[i]->operator()(state,action) - pval;
        }
        double mixValue = this->operator ()(state,action);
        arma::vec grad = vectorize(lgMix,lgCoeff) / mixValue;
        return grad;

    }

    virtual arma::mat diff2log(typename state_type<StateC>::const_type_ref state, typename action_type<ActionC>::const_type_ref action) override
    {
        //TODO [IMPORTANT] IMPLEMENT!
        return arma::mat();
    }

protected:
    virtual arma::vec computeParameters(const arma::vec& preferences)
    {
        unsigned int n = preferences.n_elem;
        arma::vec expW(n + 1);
        expW(arma::span(0, n - 1)) = preferences;
        expW(n) = 1 - arma::sum(preferences);
        return expW;
    }

protected:
    std::vector<DifferentiablePolicy<ActionC,StateC>*> mixture;
    arma::vec mixtureWeights;
};

/**
 * \f[ M(x) = \sum_{i=1}^N \alpha_i \pi_i(x) \f]
 * but \f$ \sum_{i=1}^{N} \alpha_i = 1 \f$
 * then
 * \f[ M(x) = \sum_{i=1}^{N-1} \alpha_i \pi_i(x) + (1-\sum_{i=1}^{N-1} \alpha_i) \pi_{N}(x) \f]
 *
 * Note that
 * \f[ \alpha_i = \frac{ \exp(w_i) }{ (1 + sum_j \exp(w_j)) } \f]
 */
template<class ActionC, class StateC>
class GenericParametricLogisticMixturePolicy : public GenericParametricMixturePolicy<ActionC, StateC>
{
    typedef GenericParametricMixturePolicy<ActionC, StateC> Base;
    using Base::mixture;
    using Base::mixtureWeights;

public:
    GenericParametricLogisticMixturePolicy(std::vector<DifferentiablePolicy<ActionC, StateC>*>& mixture)
        : GenericParametricMixturePolicy<ActionC, StateC>(mixture)
    {
    }

    GenericParametricLogisticMixturePolicy(std::vector<DifferentiablePolicy<ActionC, StateC>*>& mixture, arma::vec coeff)
        : GenericParametricMixturePolicy<ActionC, StateC>(mixture, coeff)
    {
    }

    virtual ~GenericParametricLogisticMixturePolicy()
    {

    }

    // Policy interface
public:

    virtual std::string getPolicyName() override
    {
        return "GenericParametricLogisticMixturePolicy";
    }

    virtual Policy<ActionC,StateC>* clone() override
    {
        return new GenericParametricLogisticMixturePolicy(this->mixture, this->mixtureWeights);
    }

    // DifferentiablePolicy interface
public:
    arma::vec difflog(typename state_type<StateC>::const_type_ref state, typename action_type<ActionC>::const_type_ref action) override
    {
        // int nbParams = mixtureWeights.n_elem;
        unsigned int nbElem = mixture.size();
        arma::mat dTheta;
        arma::vec preferences = computeParameters(mixtureWeights, dTheta);
        arma::vec lgMix, lgCoeff(nbElem);

        for (int i = 0; i < nbElem; ++i)
        {
            arma::vec gP = preferences(i) * mixture[i]->diff(state,action);
            lgMix = arma::join_vert(lgMix, gP);
        }

        for (int i = 0; i < nbElem; ++i)
        {
            lgCoeff(i) = mixture[i]->operator()(state,action);
        }
        arma::vec dM = dTheta.t() * lgCoeff;
        double mixValue = this->operator ()(state,action);
        arma::vec grad = vectorize(lgMix, dM) / mixValue;
        return grad;



//        arma::vec mixGrad = GenericParametricMixturePolicy<ActionC,StateC>::difflog(state, action);
//        arma::mat dTheta;
//        computeParameters(mixtureWeights, dTheta);

//        std::cerr << dTheta << std::endl;
//        unsigned int nel = mixGrad.n_elem;
//        std::cerr << mixGrad.rows(nel-mixtureWeights.n_elem,nel-1) << std::endl;

//        //compute the derivative
//        arma::vec dM = dTheta.t() * mixGrad.rows(nel-mixtureWeights.n_elem,nel-1);
//        return dM;

    }

    arma::mat diff2log(typename state_type<StateC>::const_type_ref state, typename action_type<ActionC>::const_type_ref action) override
    {
        //TODO [IMPORTANT] IMPLEMENT!
        return arma::mat();
    }

private:
    arma::vec computeParameters(const arma::vec& preferences) override
    {
        unsigned int n = preferences.n_elem;
        arma::vec expW(n + 1);
        expW(arma::span(0, n - 1)) = arma::exp(preferences);
        expW(n) = 1;
        return expW / sum(expW);
    }

    static arma::vec computeParameters(const arma::vec& preferences,
                                       arma::mat& dTheta)
    {
        //Compute exponential and l1-norm
        unsigned int n = preferences.n_elem;
        arma::vec expW = arma::exp(preferences);
        double D = arma::sum(expW) + 1;

        //compute derivative
        dTheta = arma::join_vert(D * arma::diagmat(expW) - expW * expW.t(),
                                 -expW.t());
        dTheta /= D * D;

        //compute parameters
        arma::vec theta(n + 1);
        theta(arma::span(0, n - 1)) = expW;
        theta(n) = 1;
        theta /= D;

        return theta;
    }
};

}

#endif //PARAMETRICMIXTUREPOLICY_H_
