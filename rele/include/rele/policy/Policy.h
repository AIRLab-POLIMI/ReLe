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

#ifndef POLICY_H_
#define POLICY_H_

#include "rele/core/BasicsTraits.h"

#include <string>
#include <map>

namespace ReLe
{

/*!
 * A commodity alias for storing hyperparameters into a map of string/double
 */
typedef std::map<std::string, double> hyperparameters_map;

/*!
 * A commodity overloading to print hyperparameters maps
 */
inline std::ostream& operator<<(std::ostream& os, const hyperparameters_map& hyperParameters)
{
    bool first = true;

    for(auto pair : hyperParameters)
    {
        if(!first)
        {
            os << std::endl;
            first = false;
        }

        os << pair.first << ": " << pair.second;

    }

    return os;
}

/*!
 * A policy provides a distribution over the action space in each state.
 * Formally, it is defined as \f$\pi : \mathcal{X} \times \mathcal{U} \to [0,1]\f$ where
 * \f$\pi(u|x)\f$ denotes the probability of action u in state x.
 * It is now clear that one basic function is represented by the possibility
 * of evaluate the probability of an action is a state. The second functionality
 * allows to draw a random action from the action distribution in a specified state.
 * Particular policies are the deterministic policies where the action is deterministically
 * choosen in each state. In this case we can say that \f$a = \pi(x)\f$.
 *
 */
template<class ActionC, class StateC>
class Policy
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

public:
    /*!
     * Draw a random action from the distribution induced by
     * the policy in a state \f$x\f$: \f$u \sim \pi(x)\f$.
     *
     * Example:
     *
     *      DenseState x();
     *      NormalPolicy policy;
     *      DenseAction u = policy(x);
     *
     *
     * \param state the state where the policy must be evaluated
     * \return an action randomly drawn from the action distribution in the given state
     */
    virtual typename action_type<ActionC>::type operator() (typename state_type<StateC>::const_type_ref state) = 0;

    /*!
     * Evaluate the density function in the provided (state,action)-pair.
     * It computes the probability of an action in a given state: \f$p = \pi(x,u)\f$.
     *
     * Example:
     * DenseState x();
     * DenseAction u();
     * NormalPolicy policy;
     * double pr = policy(x,u);
     *
     * \param state the state where the policy must be evaluated
     * \param action the action to evaluate
     * \return the probability of the (state,action)-pair
     */
    virtual double operator() (typename state_type<StateC>::const_type_ref state, typename action_type<ActionC>::const_type_ref action) = 0;

    /*!
     * Return a unique identifier of the policy type
     * \return a string storing the policy name
     */
    virtual std::string getPolicyName() = 0;

    /*!
     * Map the name of the hyperparameters to their
     * value and return such representation.
     *
     * \return the hyperparameters of the policy
     */
    virtual hyperparameters_map getPolicyHyperparameters()
    {
        return hyperparameters_map();
    }

    /*!
     * Generate a textual representation of the status
     * of the policy. It is used to visualize the policy.
     * Such representation must be human friendly.
     *
     * \return The textual description of the policy
     */
    virtual std::string printPolicy() = 0;

    /*!
     * Generate an identical copy of the current policy
     * A new object is created based on the status of the policy
     * at the time of the call. If compared the returned policy
     * is identical to the current policy but it is stored in a
     * different area of the memory.
     *
     * \return a clone of the current policy
     */
    virtual Policy<ActionC, StateC>* clone() = 0;

    virtual ~Policy()
    {

    }

};

template<class ActionC, class StateC>
class NonParametricPolicy: public Policy<ActionC, StateC>
{

};

/*!
 * A parametric policy is a specialization of a policy where the
 * representation is based on a set of parameters. This policy
 * represents a specific instance of a family of policies.
 * Let \f$\theta \in \Theta\f$ be a parameter vector. Then a
 * parametric policy defines a parametric distribution over the
 * action space, i.e.,
 * \f[ \pi(u|x,\theta) \qquad \forall x \in \mathcal{X}.\f]
 * This means that the shape of the action distribution depends
 * both on the state \f$s\f$ and on the parameter vector \f$\theta\f$.
 *
 * Example:
 *
 *
 *      Consider a normal distribution. A parametric normal policy is a
 *      normal distribution where the mean and the standard deviation
 *      are parametrized by a vector THETA.
 *
 */
template<class ActionC, class StateC>
class ParametricPolicy: public Policy<ActionC, StateC>
{
public:
    /*!
     * Provide a way to access the parameters.
     * \return the vector of parameters \f$\theta\f$
     */
    virtual arma::vec getParameters() const = 0;

    /*!
     * Return the number of parameters
     * \return the length of the vector \f$\theta\f$
     */
    virtual const unsigned int getParametersSize() const = 0;

    /*!
     * Provide a way to modify the parameters of the policy
     * by replacing the current parameter vector with the provided one.
     *
     * \param w the new policy parameters
     */
    virtual void setParameters(const arma::vec& w) = 0;

    virtual ~ParametricPolicy()
    {

    }

};

/*!
 * A differentiable policy is a specialization of a parametric policy
 * that provides first- and second-order derivatives.
 * The derivatives are computed w.r.t. the parameter vector and it is
 * evaluated in the provided (state,action)-pair with the current policy
 * parametrization.
 * Let \f$\hat{x}, \hat{u}, \hat{\theta} \subseteq R^d\f$ be the provided state, action
 * and the current policy representation, respectively. Then the first-order
 * derivative of the policy is given by
 * \f[
 * \nabla_{\theta}\pi(\hat{u}|\hat{x},\theta)\big|_{\hat{\theta}}.
 * \f]
 *
 * Note that when the policy is deterministic the diff operator works on the
 * deterministic function, i.e.,
 * \f[
 * \nabla_{\theta} \pi(u| \theta)\big|_{\hat{\theta}}.
 * \f]
 */
template<class ActionC, class StateC>
class DifferentiablePolicy: public ParametricPolicy<ActionC, StateC>
{
public:
    /*!
     * Compute the first-order derivative of the policy function which is evaluated
     * in the provided state \f$\hat{x}\f$ and action \f$\hat{u}\f$ using the current
     * parameter vector \f$\hat{\theta}\f$:
     * \f[
     * \nabla_{\theta}\pi(\hat{u}|\hat{x},\theta)\big|_{\hat{\theta}} \in \mathbb{R}^{d}.
     * \f]
     *
     * \param state the state
     * \param action the action
     * \return the first-order derivative
     */
    virtual arma::vec diff(typename state_type<StateC>::const_type_ref state, typename action_type<ActionC>::const_type_ref action)
    {
        return (*this)(state,action) * difflog(state,action);
    }

    /*!
     * Compute the first-order derivative of the policy logarithm
     * which is evaluated in the provided state
     * \f$\hat{x}\f$ and action \f$\hat{u}\f$ using the current
     * parameter vector \f$\hat{\theta}\f$:
     * \f[
     * \nabla_{\theta}\log\pi(\hat{u}|\hat{x},\theta)\big|_{\hat{\theta}} \in \mathbb{R}^{d}.
     * \f]
     *
     * \param state the state
     * \param action the action
     * \return the first-order derivative of the policy logarithm
     */
    virtual arma::vec difflog(typename state_type<StateC>::const_type_ref state, typename action_type<ActionC>::const_type_ref action) = 0;

    //virtual arma::mat diff2(typename state_type<StateC>::const_type_ref state, typename action_type<ActionC>::const_type_ref action) = 0;

    /*!
     * Compute the second-order derivative of the policy logarithm which is
     * evaluated in the provided state \f$\hat{x}\f$ and action \f$\hat{u}\f$ using the current
     * parameter vector \f$\hat{\theta}\f$:
     * \f[
     * H_{\theta}\log\pi(\hat{u}|\hat{x},\theta)\big|_{\hat{\theta}} \in \mathbb{R}^{d \times d}.
     * \f]
     * Note that the Hessian matrix is given by
     * \f[
     * H(i,j) = \frac{\partial^2}{\partial \theta_i \partial_j} \log\pi(\hat{u}|\hat{x},\theta)\big|_{\hat{\theta}},
     * \f]
     * where i is the row and j is the column.
     *
     * \param state the state
     * \param action the action
     * \return the hessian of the policy logarithm
     */
    virtual arma::mat diff2log(typename state_type<StateC>::const_type_ref state, typename action_type<ActionC>::const_type_ref action) = 0;


    virtual ~DifferentiablePolicy()
    {

    }
};

} //end namespace

#endif /* POLICY_H_ */
