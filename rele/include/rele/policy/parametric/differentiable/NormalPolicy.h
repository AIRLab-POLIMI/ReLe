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

    virtual void calculateMeanAndStddev(const arma::vec& state);

public:

    virtual double operator() (const arma::vec& state, const arma::vec& action);

    virtual arma::vec operator() (const arma::vec& state);

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
        const arma::vec& state, const arma::vec& action);

    virtual arma::vec difflog(
        const arma::vec& state, const arma::vec& action);

    virtual arma::mat diff2log(
        const arma::vec& state, const arma::vec& action);

    inline void clearRegressor(bool clear)
    {
        clearRegressorOnExit = clear;
    }

protected:
    double mInitialStddev, mMean;
    LinearApproximator* approximator;
    bool clearRegressorOnExit;

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

    virtual void calculateMeanAndStddev(const arma::vec& state);

protected:
    LinearApproximator* stdApproximator;

};

///////////////////////////////////////////////////////////////////////////////////////
/// MVN POLICY
///////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Multivariate Normal policy with fixed covariance matrix
 *
 * This class represents a multivariate Normal policy with fixed covariance matrix
 * and linear approximation of the mean value:
 * \f[ \pi^{\theta} (a|s) = \mathcal{N}(s; \phi(s)\theta, \Sigma),\qquad
 * \forall s \in R^{n_s}, a \in R^{n_a},\f]
 * where \f$\phi(s)\f$ is an \f$(n_a \times k)\f$ matrix and
 * \f$\theta\f$ is a \f$k\f$-dimensional vector.
 */
class MVNPolicy : public DifferentiablePolicy<DenseAction, DenseState>
{
public:

    /**
     * Create an instance of the class using the given projector.
     * Covariance matrix is initialized to the unit matrix.
     * Note that the weights of the mean approximation are not
     * changed, i.e., the initial weights are specified by the
     * instance of the linear projector received as parameter.
     *
     * @brief The constructor.
     * @param projector The linear projector used for mean approximation
     */
    MVNPolicy(LinearApproximator* projector)
        : approximator(projector),
          mMean(projector->GetOutputSize(), arma::fill::zeros),
          clearRegressorOnExit(false)
    {
        int output_dim = projector->GetOutputSize();
        mCovariance.eye(output_dim,output_dim);
        mCinv = arma::inv(mCovariance);
        mCholeskyDec = arma::chol(mCovariance);
        mDeterminant = arma::det(mCovariance);
    }

    /**
     * Create an instance of the class using the given projector and
     * covariance matrix.
     *
     * Example use:
     * @code
     * LinearProjector* projector = new LinearBfsProjector(...);
     * MVNPOlicy(projector, {1,2,...});
     * @endcode
     * @brief The constructor.
     * @param projector The linear projector used for mean approximation.
     * @param initialCov The covariance matrix (\f$n_a \times n_a\f$).
     */
    MVNPolicy(LinearApproximator* projector,
              std::initializer_list<double> initialCov)
        : approximator(projector),
          mMean(projector->GetOutputSize(), arma::fill::zeros),
          clearRegressorOnExit(false)
    {
        int output_dim = projector->GetOutputSize();
        mCovariance.zeros(output_dim, output_dim);
        int row = 0, col = 0 ;
        for (double x: initialCov)
        {
            mCovariance(row, col++) = x;
            if (col == output_dim)
            {
                col = 0;
                ++row;
            }
        }
        mCinv = arma::inv(mCovariance);
        mCholeskyDec = arma::chol(mCovariance);
        mDeterminant = arma::det(mCovariance);
    }

    MVNPolicy(LinearApproximator* projector,
              double* covariance)
        : approximator(projector),
          mMean(projector->GetOutputSize(), arma::fill::zeros),
          clearRegressorOnExit(false)
    {
        int output_dim = projector->GetOutputSize();
        mCovariance.zeros(output_dim, output_dim);
        for (int i = 0; i < output_dim; ++i)
        {
            for (int j = 0; j < output_dim; ++j)
            {
                mCovariance(i,j) = covariance[i+output_dim*j];
            }
        }

        mCinv = arma::inv(mCovariance);
        mCholeskyDec = arma::chol(mCovariance);
        mDeterminant = arma::det(mCovariance);
    }

    virtual ~MVNPolicy()
    {
        if (clearRegressorOnExit == true)
        {
            delete approximator;
        }
    }


public:

    virtual double operator() (const arma::vec& state, const arma::vec& action);

    virtual arma::vec operator() (const arma::vec& state);

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
        const arma::vec& state, const arma::vec& action);

    virtual arma::vec difflog(
        const arma::vec& state, const arma::vec& action);

    virtual arma::mat diff2log(
        const arma::vec& state, const arma::vec& action);

    inline void clearRegressor(bool clear)
    {
        clearRegressorOnExit = clear;
    }

protected:

    /**
     * This function is deputed to the computatio of the mean and covariance
     * values in the given state. Moreover, the function must compute all the
     * informations required for the generation of samples from the Gaussian
     * distribution and gradient computation. In particular, the flag \a cholesky_dec
     * is used to require the computation of the cholesky decomposition of the
     * covariance matrix.
     *
     * In this base version only the mean value is updated since the covariance
     * matrix is indipendent from the state value.
     *
     * @brief Update internal state.
     * @param state The state where the policy is evaluated.
     * @param cholesky_dec A flag used to require the Cholesky decomposition of the
     * covariance matrix.
     */
    void UpdateInternalState(const arma::vec& state, bool cholesky_dec = false)
    {
        //TODO: si potrebbe togliere il flag cholesky_dec e aggiungere un controllo
        // sul puntatore dello stato. Se è uguale al ultimo non ricomputo tutto

        // compute mean vector
        mMean = approximator->operator ()(state);

    }

protected:
    arma::mat mCovariance, mCinv, mCholeskyDec;
    double mDeterminant;
    LinearApproximator* approximator;
    arma::vec mMean;
    bool clearRegressorOnExit;
};


} // end namespace ReLe
#endif // NORMALPOLICY_H
