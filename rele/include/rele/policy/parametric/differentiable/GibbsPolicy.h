#ifndef GIBBSPOLICY_H_
#define GIBBSPOLICY_H_

#include "rele/policy/Policy.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/utils/RandomGenerator.h"

#include <stdexcept>

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
    std::string getPolicyName() override
    {
        return std::string("GibbsPolicy");
    }

    hyperparameters_map getPolicyHyperparameters() override
    {
    	hyperparameters_map hyperParameters;
    	hyperParameters["tau"] = tau;
        return hyperParameters;
    }

    std::string printPolicy() override
    {
        return std::string("");
    }


    double operator() (typename state_type<StateC>::const_type_ref state,
                       const unsigned int& action) override
    {
        int statesize = state.size();
        int nactions = mActions.size();

        arma::vec tuple(1+statesize);
        tuple(arma::span(0, statesize-1)) = state;

        arma::vec&& distribution = computeDistribution(nactions, tuple,	statesize);
        unsigned int index = findActionIndex(action);
        return distribution[index];
    }

    unsigned int operator() (typename state_type<StateC>::const_type_ref state) override
    {
        int statesize = state.size();
        int nactions = mActions.size();

        arma::vec tuple(1+statesize);
        tuple(arma::span(0, statesize-1)) = state;


        arma::vec&& distribution = computeDistribution(nactions, tuple,	statesize);
        unsigned int idx = RandomGenerator::sampleDiscrete(distribution.begin(), distribution.end());
        return mActions.at(idx).getActionN();
    }

    virtual ParametricGibbsPolicy<StateC>* clone() override
    {
        return new  ParametricGibbsPolicy<StateC>(*this);
    }

    // ParametricPolicy interface
public:
    virtual inline arma::vec getParameters() const override
    {
        return approximator.getParameters();
    }

    virtual inline const unsigned int getParametersSize() const override
    {
        return approximator.getParametersSize();
    }

    virtual inline void setParameters(const arma::vec& w) override
    {
        approximator.setParameters(w);
    }

    // DifferentiablePolicy interface
public:
    virtual arma::vec diff(typename state_type<StateC>::const_type_ref state,
                           typename action_type<FiniteAction>::const_type_ref action) override
    {
        ParametricGibbsPolicy& pi = *this;
        return pi(state, action)*difflog(state, action);
    }

    virtual arma::vec difflog(typename state_type<StateC>::const_type_ref state,
                              typename action_type<FiniteAction>::const_type_ref action) override
    {
        int statesize = state.size();
        int nactions = mActions.size();

        arma::vec tuple(1+statesize);
        tuple(arma::span(0, statesize-1)) = state;


        arma::vec&& distribution = computeDistribution(nactions, tuple, statesize);
        Features& phi = approximator.getFeatures();
        arma::mat features(phi.rows(), nactions);

        for (unsigned int k = 0; k < nactions; k++)
        {
            tuple[statesize] = mActions[k].getActionN();
            features.col(k) = phi(tuple) / tau;
        }

        unsigned int index = findActionIndex(action);

        return features.col(index) - features*distribution;
    }

    virtual arma::mat diff2log(typename state_type<StateC>::const_type_ref state,
                               typename action_type<FiniteAction>::const_type_ref action) override
    {
        //TODO IMPLEMENT
        return arma::mat();
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
            den += val;
            distribution[k] = val;
        }

        // check extreme cases (if some action is nan or infinite)
        arma::uvec q_nf = arma::find_nonfinite(distribution);
        if (q_nf.n_elem > 0)
        {
            arma::uvec q_f = arma::find_finite(distribution);
            distribution(q_f).zeros();
            distribution(q_nf).ones();
            den = q_nf.n_elem;
        }

        distribution /= den;
//        std::cout << distribution.t();
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
    std::string getPolicyName() override
    {
        return std::string("ParametricGibbsPolicyAllPref");
    }

    hyperparameters_map getPolicyHyperparameters() override
    {
    	hyperparameters_map hyperParameters;
    	hyperParameters["tau"] = tau;
    	return hyperParameters;
    }

    std::string printPolicy() override
    {
        return std::string("");
    }


    double operator() (typename state_type<StateC>::const_type_ref state,
                       const unsigned int& action) override
    {
        int statesize = state.size();
        int nactions = mActions.size();

        arma::vec tuple(1+statesize);
        tuple(arma::span(0, statesize-1)) = state;

        arma::vec&& distribution = computeDistribution(nactions, tuple,	statesize);
        unsigned int index = findActionIndex(action);
        return distribution[index];
    }

    unsigned int operator() (typename state_type<StateC>::const_type_ref state) override
    {
        int statesize = state.size();
        int nactions = mActions.size();

        arma::vec tuple(1+statesize);
        tuple(arma::span(0, statesize-1)) = state;


        arma::vec&& distribution = computeDistribution(nactions, tuple,	statesize);
        unsigned int idx = RandomGenerator::sampleDiscrete(distribution.begin(), distribution.end());
        return mActions.at(idx).getActionN();
    }

    virtual ParametricGibbsPolicyAllPref<StateC>* clone() override
    {
        return new  ParametricGibbsPolicyAllPref<StateC>(*this);
    }

    // ParametricPolicy interface
public:
    virtual inline arma::vec getParameters() const override
    {
        return approximator.getParameters();
    }

    virtual inline const unsigned int getParametersSize() const override
    {
        return approximator.getParametersSize();
    }

    virtual inline void setParameters(const arma::vec& w) override
    {
        approximator.setParameters(w);
    }

    // DifferentiablePolicy interface
public:
    virtual arma::vec diff(typename state_type<StateC>::const_type_ref state,
                           typename action_type<FiniteAction>::const_type_ref action) override
    {
        ParametricGibbsPolicyAllPref& pi = *this;
        return pi(state, action)*difflog(state, action);
    }

    virtual arma::vec difflog(typename state_type<StateC>::const_type_ref state,
                              typename action_type<FiniteAction>::const_type_ref action) override
    {
        int statesize = state.size();
        int nactions = mActions.size();

        arma::vec tuple(1+statesize);
        tuple(arma::span(0, statesize-1)) = state;


        arma::vec&& distribution = computeDistribution(nactions, tuple, statesize);
        Features& phi = approximator.getFeatures();
        arma::mat features(phi.rows(), nactions);

        for (unsigned int k = 0; k < nactions; k++)
        {
            tuple[statesize] = mActions[k].getActionN();
            features.col(k) = phi(tuple)/ tau;
        }

        unsigned int index = findActionIndex(action);

        return features.col(index) - features*distribution;
    }

    virtual arma::mat diff2log(typename state_type<StateC>::const_type_ref state,
                               typename action_type<FiniteAction>::const_type_ref action) override
    {
        //TODO IMPLEMENT
        return arma::mat();
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

        // check extreme cases (if some action is nan or infinite)
        arma::uvec q_nf = arma::find_nonfinite(distribution);
        if (q_nf.n_elem > 0)
        {
            arma::uvec q_f = arma::find_finite(distribution);
            distribution(q_f).zeros();
            distribution(q_nf).ones();
            den = q_nf.n_elem;
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
