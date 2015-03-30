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
                          LinearApproximator* projector, double inverseTemp) :
        mActions(actions), distribution(actions.size(),0),
        approximator(projector), clearRegressorOnExit(false),
        inverseTemperature(inverseTemp)
    {
    }

    virtual ~ParametricGibbsPolicy()
    {
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
//            AbstractBasisMatrix& basis = approximator->getBasis();
//            std::cout << basis(tuple).t();

            arma::vec preference = (*approximator)(tuple);
//            std::cout << preference << std::endl;
            double val = exp(inverseTemperature*preference[0]);
//            std::cout << val << std::endl;
            den += val;
            distribution[count++] = val;
        }
        distribution[nactions-1] = 1.0;


        for (unsigned int k = 0, ke = nactions; k < ke; ++k)
        {
            if ((isnan(distribution[k]))||isinf(distribution[k]))
            {
                distribution[k] = 1;
            }
            else
            {
                distribution[k] /= den;
            }
//            std::cout << distribution[k] << " ";
        }
//        std::cout << std::endl;

        unsigned int idx = RandomGenerator::sampleDiscrete(distribution.begin(), distribution.end());
        return mActions.at(idx).getActionN();
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
    	//TODO Implement this method
    	return arma::vec();
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
    std::vector<double> distribution;
    double inverseTemperature;
    LinearApproximator* approximator;
    bool clearRegressorOnExit;
};

}
#endif // GIBBSPOLICY_H_
