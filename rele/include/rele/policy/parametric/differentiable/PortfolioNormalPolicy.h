#ifndef PORTFOLIONORMALPOLICY_H
#define PORTFOLIONORMALPOLICY_H

#include "Policy.h"
#include "LinearApproximator.h"
#include "ArmadilloPDFs.h"

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// PORTFOLIO NORMAL POLICY
///////////////////////////////////////////////////////////////////////////////////////

/**
 * Univariate normal policy with fixed standard deviation
 */
class PortfolioNormalPolicy: public DifferentiablePolicy<FiniteAction, DenseState>
{
public:
    PortfolioNormalPolicy(const double& epsilon, Features& phi) :
        epsilon(epsilon),
        approximator(phi)
    {
    }

    virtual ~PortfolioNormalPolicy()
    {

    }

public:

    virtual double operator()(const arma::vec& state,
                              typename action_type<FiniteAction>::const_type_ref action);

    virtual unsigned int operator()(const arma::vec& state);


    virtual inline std::string getPolicyName()
    {
        return "PortfolioNormalPolicy";
    }
    virtual inline std::string getPolicyHyperparameters()
    {
        return "";
    }
    virtual inline std::string printPolicy()
    {
        return "";
    }

    virtual PortfolioNormalPolicy* clone()
    {
        return new  PortfolioNormalPolicy(*this);
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
    virtual inline void setParameters(arma::vec &w)
    {
        approximator.setParameters(w);
    }

    // DifferentiablePolicy interface
public:
    virtual arma::vec diff(const arma::vec& state,
                           typename action_type<FiniteAction>::const_type_ref action);

    virtual arma::vec difflog(const arma::vec& state,
                              typename action_type<FiniteAction>::const_type_ref action);

    virtual arma::mat diff2log(const arma::vec& state,
                               typename action_type<FiniteAction>::const_type_ref action);

protected:
    double epsilon;
    LinearApproximator approximator;
};

} // end namespace ReLe
#endif // PORTFOLIONORMALPOLICY_H
