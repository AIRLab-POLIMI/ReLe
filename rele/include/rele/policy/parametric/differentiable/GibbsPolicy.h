#ifndef GIBBSPOLICY_H_
#define GIBBSPOLICY_H_

#include "Policy.h"
#include "regressors/LinearApproximator.h"
#include "RandomGenerator.h"

#include <stdexcept>

//#define DEBUG_GIBBS

namespace ReLe
{


template<class StateC>
class ParametricGibbsPolicy : public DifferentiablePolicy<FiniteAction, StateC>
{
public:

    ParametricGibbsPolicy(std::vector<FiniteAction> actions,
                          Features& phi, double temperature) :
        mActions(actions), distribution(actions.size(),0),
        approximator(phi), tau(temperature)
    {
    }

    virtual ~ParametricGibbsPolicy()
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
        return std::string("GibbsPolicy");
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

    virtual ParametricGibbsPolicy<StateC>* clone()
    {
        return new  ParametricGibbsPolicy<StateC>(*this);
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
        ParametricGibbsPolicy& pi = *this;
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
            features.col(k) = phi(tuple) / tau;
        }

        unsigned int index = findActionIndex(action);

        return features.col(index) - features*distribution;
    }

private:
    arma::vec computeDistribution(int nactions, arma::vec tuple, int statesize)
    {
        int na_red = nactions - 1;
        arma::vec distribution(nactions);
        distribution(na_red) = 1.0; //last action is valued 1.0
        double den = 1.0; //set the value of the last action to the den
        for (unsigned int k = 0; k < na_red; k++)
        {
            tuple[statesize] = mActions[k].getActionN();
            arma::vec preference = approximator(tuple);
            double val = exp(preference[0] / tau);
            if (isnan(val) || isinf(val))
            {
                val = arma::datum::inf;
                std::cerr << "Gibbs: found inf or nan element in distribution." << std::endl;
            }
            den += val;
            distribution[k] = val;
        }

        // check extreme cases (if some action is nan or infinite
        arma::uvec q_inf = arma::find(distribution == arma::datum::inf);
        if (q_inf.n_elem > 0)
        {
            //get other elements
            std::cerr << "distribution contains infinite elements: " << distribution.t();
            arma::uvec q_not_inf = arma::find(distribution != arma::datum::inf);
            distribution.elem(q_inf).ones();
            den = q_inf.n_elem;
            distribution.elem(q_not_inf).zeros();
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

protected:
    std::vector<FiniteAction> mActions;
    std::vector<double> distribution;
    double tau;
    LinearApproximator approximator;

};

template<class StateC>
class ParametricGibbsPolicyAllPref : public DifferentiablePolicy<FiniteAction, StateC>
{
public:

    ParametricGibbsPolicyAllPref(std::vector<FiniteAction> actions,
                   Features& phi, double temperature) :
        mActions(actions), distribution(actions.size(),0),
        approximator(phi), tau(temperature)
    {
    }

    virtual ~ParametricGibbsPolicyAllPref()
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
        return std::string("ParametricGibbsPolicyAllPref");
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

    virtual ParametricGibbsPolicyAllPref<StateC>* clone()
    {
        return new  ParametricGibbsPolicyAllPref<StateC>(*this);
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
        ParametricGibbsPolicyAllPref& pi = *this;
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
            if (isnan(val) || isinf(val))
            {
                val = arma::datum::inf;
                std::cerr << "Gibbs: found inf or nan element in distribution." << std::endl;
            }
            den += val;
            distribution[k] = val;
        }

        // check extreme cases (if some action is nan or infinite
        arma::uvec q_inf = arma::find(distribution == arma::datum::inf);
        if (q_inf.n_elem > 0)
        {
            //get other elements
            std::cerr << "distribution contains infinite elements: " << distribution.t();
            arma::uvec q_not_inf = arma::find(distribution != arma::datum::inf);
            distribution.elem(q_inf).ones();
            den = q_inf.n_elem;
            distribution.elem(q_not_inf).zeros();
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
#endif // GIBBSPOLICY_H_
