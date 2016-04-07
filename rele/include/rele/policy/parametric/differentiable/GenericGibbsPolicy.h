#ifndef GIBBSPOLICY_H_
#define GIBBSPOLICY_H_

#include "rele/policy/Policy.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/utils/RandomGenerator.h"

#include <stdexcept>

namespace ReLe
{

/*!
 * When the action space is finite, a popular choice is to use the Gibbs policy,
 * a parametric differentiable policy.
 * Here we consider a parametric representation of the utility (energy) function:
 * \f[\pi(u|x,\theta) = \frac{\exp(Q(x,u,\theta)/\tau}{\sum_{u'\in mathcal{U}} \exp(Q(x,u',\theta))/\tau},\quad x \in \mathcal{X}, u \in \mathcal{U},\f]
 * where \f$Q : \mathcal{X} \times \mathcal{U} \times \Theta \to \mathbb{R}\f$ is an appropriate
 * feature-extraction function and \f$\tau\f$ is the temperature. The temperature is used to
 * regulate the level of determinism of the policy.
 *
 * Since the policy must define a distribution over the actions, we can avoid to associate weights
 * to the last action since its probability can be obtained from previous actions.
 * Let \f$\overline{u}\f$ be the last action, then \f$\exp(Q(x,\overline{u},\theta)) = 0\f$ by assumption.
 * As a consequence
 * \f[\pi(u| x,\theta) = \frac{\exp(Q(x,u,\theta)/\tau}{\sum_{1+ u'\in mathcal{U}\setminus\overline{u}} \exp(Q(x,u',\theta))/\tau},\quad x \in \mathcal{X}, u \in \mathcal{U},\f]
 *
 * \see{GenericParametricGibbsPolicyAllPref}
 */
template<class StateC>
class GenericParametricGibbsPolicy : public DifferentiablePolicy<FiniteAction, StateC>
{
public:

    /**
     * Create an instance of the class using the given regressor.
     *
     * \param actions the vector of finite actions
     * \param energy the energy function \f$Q(x,u,\theta)\f$
     * \param temperature the temperature value
     */
    GenericParametricGibbsPolicy(std::vector<FiniteAction> actions,
                                 ParametricRegressor& energy, double temperature) :
        mActions(actions), distribution(actions.size(),0),
        approximator(energy), tau(temperature)
    {
    }

    virtual ~GenericParametricGibbsPolicy()
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
        return std::string("GenericParametricGibbsPolicy");
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

    virtual GenericParametricGibbsPolicy<StateC>* clone() override
    {
        return new  GenericParametricGibbsPolicy<StateC>(*this);
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
        GenericParametricGibbsPolicy& pi = *this;
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
        unsigned int index = findActionIndex(action);

        arma::vec gradient;
        if (index != nactions-1)
        {
            tuple[statesize] = mActions[index].getActionN();
            gradient = approximator.diff(tuple);
        }
        else
        {
            gradient.zeros(approximator.getParametersSize());
        }
        for (unsigned int k = 0; k < nactions - 1; k++)
        {
            tuple[statesize] = mActions[k].getActionN();
            gradient -= approximator.diff(tuple)*distribution(k);
        }

        return gradient / tau;
    }

    virtual arma::mat diff2log(typename state_type<StateC>::const_type_ref state,
                               typename action_type<FiniteAction>::const_type_ref action) override
    {
        //TODO [IMPORTANT] Implement
        return arma::mat();
    }

private:
    /*!
     * Compute the distribution in the current state, i.e., it computes the
     * vector of probabilities of each action in state \f$x\f$.
     * \param nactions the finite number of actions
     * \param tuple a vector of dimension \f$n_x + 1\f$. The first \f$n_x\f$
     * elements contain the current state \f$x\f$, while the action (0-nactions) is filled
     * by the function
     * \param statesize the dimension of the state (\f$n_x\f$)
     * \return A vector representing the action distribution in state \f$x\f$
     *
     * Example:
     *                  int statesize = state.size();
     *                  int nactions = mActions.size();
     *                  arma::vec tuple(1+statesize);
     *                  tuple(arma::span(0, statesize-1)) = state;
     *                  arma::vec distribution = computeDistribution(nactions, tuple,	statesize);
     */
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

    /*!
     * Look for an action in the action list
     * \param action the action to be searched
     * \return return the action number
     */
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
    ParametricRegressor& approximator;

};


/*!
 * When the action space is finite, a popular choice is to use the Gibbs policy,
 * a parametric differentiable policy.
 * Here we consider a parametric representation of the utility (energy) function:
 * \f[\pi(u|x,\theta) = \frac{\exp(Q(x,u,\theta)/\tau}{\sum_{u'\in mathcal{U}} \exp(Q(x,u',\theta))/\tau},\quad x \in \mathcal{X}, u \in \mathcal{U},\f]
 * where \f$Q : \mathcal{X} \times \mathcal{U} \times \Theta \to \mathbb{R}\f$ is an appropriate
 * feature-extraction function and \f$\tau\f$ is the temperature. The temperature is used to
 * regulate the level of determinism of the policy.
 */
template<class StateC>
class GenericParametricGibbsPolicyAllPref : public DifferentiablePolicy<FiniteAction, StateC>
{
public:

    /**
     * Create an instance of the class using the given regressor.
     *
     * \param actions the vector of finite actions
     * \param energy the energy function \f$Q(x,u,\theta)\f$
     * \param temperature the temperature value
     */
    GenericParametricGibbsPolicyAllPref(std::vector<FiniteAction> actions,
                                        ParametricRegressor& energy, double temperature) :
        mActions(actions), distribution(actions.size(),0),
        approximator(energy), tau(temperature)
    {
    }

    virtual ~GenericParametricGibbsPolicyAllPref()
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
        return std::string("GenericParametricGibbsPolicyAllPref");
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

    virtual GenericParametricGibbsPolicyAllPref<StateC>* clone() override
    {
        return new  GenericParametricGibbsPolicyAllPref<StateC>(*this);
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
        GenericParametricGibbsPolicyAllPref& pi = *this;
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
        unsigned int index = findActionIndex(action);
        tuple[statesize] = mActions[index].getActionN();
        arma::vec&& diffsa = approximator.diff(tuple);
        arma::vec sumdiff(diffsa.n_elem, arma::fill::zeros);
        for (unsigned int k = 0; k < nactions; k++)
        {
            tuple[statesize] = mActions[k].getActionN();
            sumdiff += approximator.diff(tuple)*distribution(k);
        }

        return (diffsa - sumdiff) / tau;
    }

    virtual arma::mat diff2log(typename state_type<StateC>::const_type_ref state,
                               typename action_type<FiniteAction>::const_type_ref action) override
    {
        //TODO [IMPORTANT] Implement
        return arma::mat();
    }


private:

    /*!
     * Compute the distribution in the current state, i.e., it computes the
     * vector of probabilities of each action in state \f$x\f$.
     * \param nactions the finite number of actions
     * \param tuple a vector of dimension \f$n_x + 1\f$. The first \f$n_x\f$
     * elements contain the current state \f$x\f$, while the action (0-nactions) is filled
     * by the function
     * \param statesize the dimension of the state (\f$n_x\f$)
     * \return A vector representing the action distribution in state \f$x\f$
     *
     * Example:
     *                  int statesize = state.size();
     *                  int nactions = mActions.size();
     *                  arma::vec tuple(1+statesize);
     *                  tuple(arma::span(0, statesize-1)) = state;
     *                  arma::vec distribution = computeDistribution(nactions, tuple,	statesize);
     */
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

    /*!
     * Look for an action in the action list
     * \param action the action to be searched
     * \return return the action number
     */
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
    ParametricRegressor& approximator;
};

}
#endif // GIBBSPOLICY_H_
