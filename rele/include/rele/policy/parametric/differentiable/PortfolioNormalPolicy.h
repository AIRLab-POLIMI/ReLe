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
    PortfolioNormalPolicy(const double& epsilon, LinearApproximator* projector) :
        epsilon(epsilon), mMean(0.0),
        approximator(projector), clearRegressorOnExit(false)
    {
    }

    virtual ~PortfolioNormalPolicy()
    {
        if (clearRegressorOnExit == true)
        {
            delete approximator;
        }
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

    // ParametricPolicy interface
public:
    virtual inline const arma::vec &getParameters() const
    {
        return approximator->getParameters();
    }
    virtual inline const unsigned int getParametersSize() const
    {
        return approximator->getParameters().n_elem;
    }
    virtual inline void setParameters(arma::vec &w)
    {
        approximator->setParameters(w);
    }

    // DifferentiablePolicy interface
public:
    virtual arma::vec diff(const arma::vec& state,
                           typename action_type<FiniteAction>::const_type_ref action);

    virtual arma::vec difflog(const arma::vec& state,
                              typename action_type<FiniteAction>::const_type_ref action);

    virtual arma::mat diff2log(const arma::vec& state,
                               typename action_type<FiniteAction>::const_type_ref action);

    inline void clearRegressor(bool clear)
    {
        clearRegressorOnExit = clear;
    }

protected:
    double epsilon, mMean;
    LinearApproximator* approximator;
    bool clearRegressorOnExit;
};

} // end namespace ReLe
#endif // PORTFOLIONORMALPOLICY_H
