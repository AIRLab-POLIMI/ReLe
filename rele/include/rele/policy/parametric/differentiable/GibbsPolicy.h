#ifndef GIBBSPOLICY_H_
#define GIBBSPOLICY_H_

#include "Policy.h"
#include "LinearApproximator.h"
#include "RandomGenerator.h"

namespace ReLe
{


template<class StateC>
class ParametricGibbsPolicy : public DifferentiablePolicy<FiniteAction, StateC>
{
public:

    ParametricGibbsPolicy(std::vector<FiniteAction> actions,
                          LinearApproximator* projector) :
        mActions(actions), distribution(new double[actions.size()]),
        approximator(projector), clearRegressorOnExit(false)
    {
    }

    virtual ~ParametricGibbsPolicy()
    {
        delete [] distribution;
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
        for (unsigned int i = 0; i < statesize; ++i)
        {
            tuple[i] = state[i];
        }
        double den = 1.0;
        for (unsigned int k = 0, ke = nactions - 1; k < ke; ++k)
        {
            tuple[statesize] = mActions[k].getActionN();
            arma::vec preference = (*approximator)(tuple);
            den += exp(inverseTemperature*preference[0]);
        }

        tuple[statesize] = action;

        double num = 1.0;
        if (action != mActions[nactions - 1].getActionN())
        {
            arma::vec preference = (*approximator)(tuple);
            num = exp(inverseTemperature*preference[0]);
        }

        return num/den;
    }

    unsigned int operator() (typename state_type<StateC>::const_type_ref state)
    {
        double den = 1.0;
        int count = 0, nactions = mActions.size();

        int statesize = state.size();
        arma::vec tuple(1+statesize);
        for (unsigned int i = 0; i < statesize; ++i)
        {
            tuple[i] = state[i];
        }

        for (unsigned int k = 0, ke = nactions - 1; k < ke; ++k)
        {
            tuple[statesize] = mActions[k].getActionN();
            arma::vec preference = (*approximator)(tuple);
            double val = exp(inverseTemperature*preference[0]);
            den += val;
            distribution[count++] = val;
        }
        distribution[nactions-1] = 1.0;

        double random = RandomGenerator::sampleUniform(0,1);
        double sum = 0.0, pval = random*den;
        for (unsigned int i = 0, ie = nactions; i < ie; ++i)
        {
            sum += distribution[i];
            if (sum >= pval)
            {
                return mActions.at(i).getActionN();
            }
        }
        return mActions.at(nactions - 1).getActionN();
    }

    // ParametricPolicy interface
public:
    virtual inline const arma::vec& getParameters() const
    {
        return approximator->getParameters();
    }
    virtual inline const unsigned int getParametersSize() const
    {
        return approximator->getParameters().n_elem;
    }
    virtual inline void setParameters(arma::vec& w)
    {
        approximator->setParameters(w);
    }

    // DifferentiablePolicy interface
public:
    virtual arma::vec diff(typename state_type<StateC>::const_type_ref state,
                           typename action_type<FiniteAction>::const_type_ref action)
    {

    }

    virtual arma::vec difflog(typename state_type<StateC>::const_type_ref state,
                              typename action_type<FiniteAction>::const_type_ref action)
    {
        double IT = inverseTemperature; //inverse temperature

        // Compute the sum of all the preferences
        double sumexp = 0;
        arma::mat sumpref(this->getParametersSize(),1,arma::fill::zeros); // sum of the preferences
        unsigned int nactions = mActions.size();

        int statesize = state.size();
        arma::vec tuple(1+statesize);
        for (unsigned int i = 0; i < statesize; ++i)
        {
            tuple[i] = state[i];
        }

        AbstractBasisMatrix& basis = approximator->getBasis();
        for (unsigned int k = 0, ke = nactions - 1; k < ke; ++k)
        {
            tuple[statesize] = mActions[k].getActionN();
            arma::vec pref = (*approximator)(tuple);
            arma::mat loc_phi = basis(tuple);
            double val = exp(IT*pref[0]);
            distribution[k] = val;
            sumexp = sumexp + val;
            sumpref = sumpref + IT*loc_phi*distribution[k];
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
            gradient = IT*loc_phi - sumpref;
        }
        return gradient;
    }

protected:
    std::vector<FiniteAction> mActions;
    double* distribution, inverseTemperature;
    LinearApproximator* approximator;
    bool clearRegressorOnExit;
};

}
#endif // GIBBSPOLICY_H_
