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
                       typename action_type<FiniteAction>::const_type_ref action)
    {
        int statesize = state.size(), nactions = mActions.size();
        arma::vec tuple(1+statesize);
        const std::vector<bool>& mask = this->getMask(state, nactions);


        for (unsigned int i = 0; i < statesize; ++i)
        {
            tuple[i] = state[i];
        }
        double den = 1.0;
        for (unsigned int k = 0, ke = nactions - 1; k < ke; ++k)
        {
            tuple[statesize] = mActions[k].getActionN();
            arma::vec preference = approximator(tuple);
            den += mask[k]*exp(preference[0]/tau);
        }

        tuple[statesize] = action;

        double num = mask[action]*1.0;
        if (action != mActions[nactions - 1].getActionN())
        {
            arma::vec preference = approximator(tuple);
            num = mask[action]*exp(preference[0]/tau);
        }

        return num/den;
    }

    unsigned int operator() (typename state_type<StateC>::const_type_ref state)
    {
        double den = 1.0;
        int count = 0, nactions = mActions.size();
        const std::vector<bool>& mask = this->getMask(state, nactions);

        int statesize = state.size();
        arma::vec tuple(1+statesize);
        for (unsigned int i = 0; i < statesize; ++i)
        {
            tuple[i] = state[i];
        }

        for (unsigned int k = 0, ke = nactions - 1; k < ke; ++k)
        {
            tuple[statesize] = mActions[k].getActionN();

            arma::vec preference = approximator(tuple);
            double val = mask[k]*exp(preference[0]/tau);
            den += val;
            distribution[count++] = val;
        }
        distribution[nactions-1] = mActions[nactions-1]*1.0;


        double new_den = 0.0;
        for (unsigned int k = 0, ke = nactions; k < ke; ++k)
        {
#ifdef DEBUG_GIBBS
            double val = den;
            if (isinf(val))
            {
                throw std::runtime_error("Distribution is infinite");
            }
            else if(isnan(val))
            {
                throw std::runtime_error("Distribution is NaN");
            }
#endif
            distribution[k] /= den;
            if (isnan(distribution[k]) || isinf(distribution[k]))
            {
                distribution[k] = 1;
            }
            new_den += distribution[k];
        }

        if (!(abs(new_den - 1.0) < 1e-5))
        {
            for (unsigned int k = 0, ke = nactions; k < ke; ++k)
            {
                distribution[k] /= new_den;
            }
        }

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
        return (*this)(state, action)*difflog(state, action);
    }

    virtual arma::vec difflog(typename state_type<StateC>::const_type_ref state,
                              typename action_type<FiniteAction>::const_type_ref action)
    {
        // Compute the sum of all the preferences
        double sumexp = 0;
        arma::mat sumpref(this->getParametersSize(),1,arma::fill::zeros); // sum of the preferences
        unsigned int nactions = mActions.size();
        const std::vector<bool>& mask = this->getMask(state, nactions);

        int statesize = state.size();
        arma::vec tuple(1+statesize);
        for (unsigned int i = 0; i < statesize; ++i)
        {
            tuple[i] = state[i];
        }

        Features& basis = approximator.getBasis();
        for (unsigned int k = 0, ke = nactions - 1; k < ke; ++k)
        {
            tuple[statesize] = mActions[k].getActionN();
            arma::vec pref = approximator(tuple);
            arma::mat loc_phi = basis(tuple);
            double val = mask[k]*std::min(exp(pref[0]/tau), 1e200);
            distribution[k] = val;

#ifdef DEBUG_GIBBS
            if (isinf(val))
            {
                throw std::runtime_error("Distribution is infinite");
            }
            else if(isnan(val))
            {
                throw std::runtime_error("Distribution is NaN");
            }
#endif

            sumexp = sumexp + distribution[k];
            sumpref = sumpref + loc_phi*distribution[k]/tau;
        }
        sumexp = sumexp + 1;
        sumpref = sumpref / sumexp;

        arma::vec gradient;
        if (action == mActions[mActions.size()-1].getActionN())
            gradient = -sumpref;
        else
        {
            tuple[statesize] = action;
            arma::mat loc_phi = basis(tuple);
            gradient = loc_phi/tau - sumpref;
        }
        return gradient;
    }

protected:
    std::vector<FiniteAction> mActions;
    std::vector<double> distribution;
    double tau;
    LinearApproximator approximator;

};

}
#endif // GIBBSPOLICY_H_
