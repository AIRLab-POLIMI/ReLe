/*
 * IdentityPolicy.h
 *
 *  Created on: Dec 22, 2015
 *      Author: francesco
 */

#ifndef IDENTITYPOLICY_H_
#define IDENTITYPOLICY_H_

namespace ReLe
{

template<class StateC>
class IdentityPolicy: public ParametricPolicy<FiniteAction, StateC>
{
public:
    virtual ~IdentityPolicy()
    {
    }

public:
    virtual unsigned int operator() (typename state_type<StateC>::const_type_ref state) override
    {
        return action;
    }

    virtual double operator() (typename state_type<StateC>::const_type_ref state, const unsigned int& action) override
    {
        return this->action == action;
    }

    virtual std::string getPolicyName() override
    {
        return "Identity Policy";
    }

    virtual std::string printPolicy() override
    {
        std::stringstream ss;
        ss << "Identity Policy " << action << std::endl;

        return ss.str();
    }

    virtual Policy<FiniteAction, StateC>* clone() override
    {
        return new IdentityPolicy(*this);
    }

public:
    virtual arma::vec getParameters() const override
    {
        arma::vec p(1);
        p(0) = action;
        return p;
    }

    virtual const unsigned int getParametersSize() const override
    {
        return 1;
    }

    virtual void setParameters(const arma::vec& w) override
    {
        assert(w.n_elem == 1);
        action = w(0);
    }

private:
    unsigned int action;
};

template<class StateC>
class DenseIdentityPolicy: public ParametricPolicy<FiniteAction, StateC>
{
public:
    DenseIdentityPolicy(unsigned int size):
        action(size, arma::fill::zeros)
    {
    }

    virtual ~DenseIdentityPolicy()
    {
    }

public:
    virtual arma::vec operator() (typename state_type<StateC>::const_type_ref state) override
    {
        return action;
    }

    virtual double operator() (typename state_type<StateC>::const_type_ref state, const arma::vec& action) override
    {
        return arma::all(this->action == action);
    }

    virtual std::string getPolicyName() override
    {
        return "Dense Identity Policy";
    }

    virtual std::string printPolicy() override
    {
        std::stringstream ss;
        ss << "Dense Identity Policy " << action.t();

        return ss.str();
    }

    virtual Policy<DenseAction, StateC>* clone() override
    {
        return new DenseIdentityPolicy(*this);
    }

public:
    virtual arma::vec getParameters() const override
    {
        return action;
    }

    virtual const unsigned int getParametersSize() const override
    {
        return action.n_elem;
    }

    virtual void setParameters(const arma::vec& w) override
    {
        assert(w.n_elem == action.n_elem);
        action = w;
    }

private:
    arma::vec action;
};

}
#endif /* IDENTITYPOLICY_H_ */
