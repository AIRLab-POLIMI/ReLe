#ifndef NORMALPOLICY_H
#define NORMALPOLICY_H

#include "Policy.h"
#include "LinearApproximator.h"
#include "ArmadilloPDFs.h"

#define NORMALP_NAME "NormalPolicy"

namespace ReLe
{


///////////////////////////////////////////////////////////////////////////////////////
/// NORMAL POLICY
///////////////////////////////////////////////////////////////////////////////////////

/**
 * Univariate normal policy with fixed standard deviation
 */
class NormalPolicy : public DifferentiablePolicy<DenseAction, DenseState>
{
public:
    NormalPolicy(const double& initialStddev,
                 LinearApproximator* projector) :
        mInitialStddev(initialStddev), mMean(0.0),
        approximator(projector)
    { }

    virtual ~NormalPolicy()
    {
        if (clearRegressorOnExit == true)
        {
            delete approximator;
        }
    }

protected:

    virtual void calculateMeanAndStddev(const DenseState& state);

public:

    virtual double operator() (const DenseState& state, const DenseAction& action);

    virtual DenseAction operator() (const DenseState& state);

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
    virtual arma::vec diff(
        const DenseState& state, const DenseAction& action);

    virtual arma::vec difflog(
        const DenseState& state, const DenseAction& action);

    virtual arma::mat diff2log(
        const DenseState& state, const DenseAction& action);

    inline void clearRegressor(bool clear)
    {
        clearRegressorOnExit = clear;
    }

protected:
    double mInitialStddev, mMean;
    LinearApproximator* approximator;
    bool clearRegressorOnExit;
    DenseAction todoAction;

};


///////////////////////////////////////////////////////////////////////////////////////
/// NORMAL POLICY WITH STATE DEPENDANT STDDEV (STD is not a parameter to be learned)
///////////////////////////////////////////////////////////////////////////////////////

/**
 * Univariate normal policy with state dependant standard deviation
 * \f[
 * \pi(a,s) = N(\theta^{T}\phi(s), k^{T}\phi(s)),
 * \f]
 * where \f$\theta\f$ are the parameters to be learned, \f$k\f$ is fixed.
 * An equivalent formulation is
 * \f[
 * \pi(a|s) = \left(\theta + \epsilon \right)^{T} \phi(s),
 * \f]
 * where \f$\epsilon \sim N(0, k^{T}\phi(s))\f$.
 */
class NormalStateDependantStddevPolicy : public NormalPolicy
{

public:
    NormalStateDependantStddevPolicy(LinearApproximator* projector,
                                     LinearApproximator* stdprojector) :
        NormalPolicy(1, projector),
        stdApproximator(stdprojector)
    { }

    virtual ~NormalStateDependantStddevPolicy()
    {
        if (clearRegressorOnExit == true)
        {
//            delete mpProjector;
            delete stdApproximator;
        }
    }

protected:

    virtual void calculateMeanAndStddev(const DenseState& state);

protected:
    LinearApproximator* stdApproximator;

};



} // end namespace ReLe
#endif // NORMALPOLICY_H
