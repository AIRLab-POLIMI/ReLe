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

#ifndef NORMALPOLICY_H
#define NORMALPOLICY_H

#include "Policy.h"
#include "regressors/LinearApproximator.h"
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
class NormalPolicy: public DifferentiablePolicy<DenseAction, DenseState>
{
public:
    NormalPolicy(const double initialStddev, Features& phi) :
        mInitialStddev(initialStddev), mMean(0.0),
        approximator(phi)
    {
        assert(phi.cols() == 1);
        assert(phi.rows() >= 1);
    }

    virtual ~NormalPolicy()
    {

    }

protected:

    virtual void calculateMeanAndStddev(const arma::vec& state);

public:

    virtual double operator()(const arma::vec& state, const arma::vec& action);

    virtual arma::vec operator()(const arma::vec& state);


    virtual inline std::string getPolicyName()
    {
        return "NormalPolicy";
    }
    virtual inline std::string getPolicyHyperparameters()
    {
        return "";
    }
    virtual inline std::string printPolicy()
    {
        return "";
    }

    virtual NormalPolicy* clone()
    {
        return new  NormalPolicy(*this);
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
    virtual arma::vec diff(const arma::vec& state, const arma::vec& action);

    virtual arma::vec difflog(const arma::vec& state, const arma::vec& action);

    virtual arma::mat diff2log(const arma::vec& state, const arma::vec& action);

protected:
    double mInitialStddev, mMean;
    LinearApproximator approximator;
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
class NormalStateDependantStddevPolicy: public NormalPolicy
{

public:
    NormalStateDependantStddevPolicy(Features& phi,
                                     Features& stdPhi, arma::vec& stdDevParameters) :
        NormalPolicy(1, phi), stdApproximator(stdPhi)
    {
        stdApproximator.setParameters(stdDevParameters);
    }

    virtual ~NormalStateDependantStddevPolicy()
    {

    }


    virtual inline std::string getPolicyName()
    {
        return "NormalStateDependantStddevPolicy";
    }

    virtual inline std::string getPolicyHyperparameters()
    {
        return "";
    }

    virtual inline std::string printPolicy()
    {
        return "";
    }

    virtual NormalStateDependantStddevPolicy* clone()
    {
        return new  NormalStateDependantStddevPolicy(*this);
    }

protected:

    virtual void calculateMeanAndStddev(const arma::vec& state);

protected:
    LinearApproximator stdApproximator;

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
class MVNPolicy: public DifferentiablePolicy<DenseAction, DenseState>
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
    MVNPolicy(Features& phi) :
        approximator(phi),
        mMean(approximator.getOutputSize(), arma::fill::zeros)
    {
        int output_dim = approximator.getOutputSize();
        mCovariance.eye(output_dim, output_dim);
        mCinv = arma::inv(mCovariance);
        mCholeskyDec = arma::chol(mCovariance);
        mDeterminant = arma::det(mCovariance);
    }

    MVNPolicy(Features& phi, arma::mat& covariance) :
        approximator(phi),
        mMean(approximator.getOutputSize(), arma::fill::zeros),
        mCovariance(covariance)
    {
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
    MVNPolicy(Features& phi,
              std::initializer_list<double> initialCov) :
        approximator(phi),
        mMean(approximator.getOutputSize(), arma::fill::zeros)
    {
        int output_dim = approximator.getOutputSize();
        mCovariance.zeros(output_dim, output_dim);
        int row = 0, col = 0;
        for (double x : initialCov)
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

    MVNPolicy(Features& phi, double* covariance) :
        approximator(phi),
        mMean(approximator.getOutputSize(), arma::fill::zeros)
    {
        int output_dim = approximator.getOutputSize();
        mCovariance.zeros(output_dim, output_dim);
        for (int i = 0; i < output_dim; ++i)
        {
            for (int j = 0; j < output_dim; ++j)
            {
                mCovariance(i, j) = covariance[i + output_dim * j];
            }
        }

        mCinv = arma::inv(mCovariance);
        mCholeskyDec = arma::chol(mCovariance);
        mDeterminant = arma::det(mCovariance);
    }

    virtual ~MVNPolicy()
    {

    }

    virtual inline std::string getPolicyName()
    {
        return "MVNPolicy";
    }
    virtual inline std::string getPolicyHyperparameters()
    {
        return "";
    }
    virtual inline std::string printPolicy()
    {
        return "";
    }

public:

    virtual double operator()(const arma::vec& state, const arma::vec& action);

    virtual arma::vec operator()(const arma::vec& state);

    virtual MVNPolicy* clone()
    {
        return new  MVNPolicy(*this);
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
    virtual arma::vec diff(const arma::vec& state, const arma::vec& action);

    virtual arma::vec difflog(const arma::vec& state, const arma::vec& action);

    virtual arma::mat diff2log(const arma::vec& state, const arma::vec& action);

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
    inline virtual void updateInternalState(const arma::vec& state, bool cholesky_dec = false)
    {
        //TODO: si potrebbe togliere il flag cholesky_dec e aggiungere un controllo
        // sul puntatore dello stato. Se è uguale al ultimo non ricomputo tutto

        // compute mean vector
        mMean = approximator(state);

    }

protected:
    arma::mat mCovariance, mCinv, mCholeskyDec;
    double mDeterminant;
    LinearApproximator approximator;
    arma::vec mMean;
};


///////////////////////////////////////////////////////////////////////////////////////
/// MVN POLICY with state dependant covariance
///////////////////////////////////////////////////////////////////////////////////////

class MVNStateDependantStddevPolicy : public MVNPolicy
{
public:
    MVNStateDependantStddevPolicy(Features& phi, Features& phiStdDev, arma::mat& stdDevW)
        : MVNPolicy(phi), phiStdDev(phiStdDev), stdDevW(stdDevW)
    {
        assert(phiStdDev.cols() == phi.cols());
        assert(stdDevW.n_rows == phiStdDev.cols());
        assert(stdDevW.n_cols == phiStdDev.rows());
    }

    virtual inline std::string getPolicyName()
    {
        return "MVNStateDependantStddevPolicy";
    }
    virtual inline std::string getPolicyHyperparameters()
    {
        return "";
    }
    virtual inline std::string printPolicy()
    {
        return "";
    }

protected:
    inline virtual void updateInternalState(const arma::vec& state, bool cholesky_dec = false)
    {
        mCovariance=stdDevW*phiStdDev(state);
        mCovariance = mCovariance*mCovariance.t();
        mCinv = arma::inv(mCovariance);
        mCholeskyDec = arma::chol(mCovariance);
        mDeterminant = arma::det(mCovariance);

        // compute mean vector
        mMean = approximator(state);
    }

protected:
    Features& phiStdDev;
    arma::mat& stdDevW;

};


///////////////////////////////////////////////////////////////////////////////////////
/// MVN POLICY with Diagonal covariance (parameters of the diagonal are stddev)
///////////////////////////////////////////////////////////////////////////////////////
class MVNDiagonalPolicy : public MVNPolicy
{
public:
    MVNDiagonalPolicy(Features& phi)
        :MVNPolicy(phi), stddevParams(approximator.getOutputSize(),arma::fill::ones)
    {
        UpdateCovarianceMatrix();
    }

    MVNDiagonalPolicy(Features& phi,
                      arma::vec stddevVector)
        :MVNPolicy(phi), stddevParams(stddevVector)
    {
        UpdateCovarianceMatrix();
    }

    virtual ~MVNDiagonalPolicy()
    {

    }

    virtual inline std::string getPolicyName()
    {
        return "MVNDiagonalPolicy";
    }
    virtual inline std::string getPolicyHyperparameters()
    {
        return "";
    }
    virtual inline std::string printPolicy()
    {
        return "";
    }

    virtual MVNDiagonalPolicy* clone()
    {
        return new  MVNDiagonalPolicy(*this);
    }

    // ParametricPolicy interface
public:
    virtual inline arma::vec getParameters() const
    {
        return arma::join_vert(approximator.getParameters(), stddevParams);
    }
    virtual inline const unsigned int getParametersSize() const
    {
        return approximator.getParametersSize() + stddevParams.n_elem;
    }
    virtual void setParameters(arma::vec &w);

    // DifferentiablePolicy interface
public:

    virtual arma::vec difflog(const arma::vec& state, const arma::vec& action);

    virtual arma::mat diff2log(const arma::vec& state, const arma::vec& action);

private:
    /**
     * Compute the covariance matrix from the logistic parameters and
     * compute the Cholesky decomposition of it.
     *
     * @brief Update the covariance matrix
     */
    void UpdateCovarianceMatrix();

protected:
    arma::vec stddevParams;
};

///////////////////////////////////////////////////////////////////////////////////////
/// MVNLogisticPolicy
///////////////////////////////////////////////////////////////////////////////////////

/**
 * This class represents a Multivariate Normal policy with
 * linearly approximated mean value and diagonal covariance matrix
 * parameterized via logistic functions:
 * \f[\pi^{\theta}(a|s) = \mathcal{N}(a;\phi(s)\rho, \Sigma^{\Omega}),\qquad
 * \forall s \in R^{n_s}, a \in R^{n_a},\f]
 * where \f$\phi(s)\f$ is an \f$(n_a \times k)\f$ matrix,
 * \f$\rho\f$ is a \f$k\f$-dimensional vector and
 * \f$\Sigma^{\Omega}\f$ is a \f$(n_a \times n_a)\f$ diagonal matrix
 * such that \f$\Sigma_{ii} = \frac{\tau}{1+e^{-\omega}}\f$.
 *
 * As a consequence, the parameter vector \f$\theta\f$ is obtained by
 * concatenation of the mean and covariance parameters:
 * \f[\rho = [K, \Omega]^{T},\f]
 * where
 * \f$\rho=[\rho_1,\dots,\rho_k]\f$ and \f$\Omega = [\omega_1, \dots,\omega_{n_a}]\f$.
 *
 *
 * @brief Multivariate Normal distribution with logistic diagonal covariance matrix
 */
class MVNLogisticPolicy : public MVNPolicy
{
protected:
    arma::vec mLogisticParams, mAsVariance;
public:

    /**
     * Create an instance of Multivariate logistic policy with the given
     * parameters. \a variance_asymptote defines the asymptotic value of the
     * logistic function used for variance approximation:
     * \f[\lim_{w \to +\infty} \frac{\tau}{1+e^{-w}} = \tau,\f]
     * where \f$\tau\f$ is equal to \a variance_asymptote.
     *
     * @brief The constructor.
     * @param projector The linear projector used for mean approximation
     * @param variance_asymptote The asymptotic value of the logistic function.
     */
    MVNLogisticPolicy(Features& phi,
                      arma::vec variance_asymptote)
        : MVNPolicy(phi),
          mLogisticParams (arma::zeros<arma::vec>(approximator.getOutputSize())),
          mAsVariance(variance_asymptote)
    {
//        unsigned int out_dim = projector->getOutputSize();
//        mCovariance.zeros(out_dim, out_dim);
        UpdateCovarianceMatrix();
    }

    MVNLogisticPolicy(Features& phi,
                      double variance_asymptote)
        : MVNPolicy(phi),
          mLogisticParams (arma::zeros<arma::vec>(approximator.getOutputSize())),
          mAsVariance(arma::ones<arma::vec>(1)*variance_asymptote)
    {
//        unsigned int out_dim = projector->getOutputSize();
//        mCovariance.zeros(out_dim, out_dim);
        UpdateCovarianceMatrix();
    }

    MVNLogisticPolicy(Features& phi,
                      arma::vec variance_asymptote,
                      arma::vec varianceparams)
        : MVNPolicy(phi),
          mLogisticParams (varianceparams),
          mAsVariance(variance_asymptote)
    {
        unsigned int out_dim = approximator.getOutputSize();
        mCovariance.zeros(out_dim, out_dim);
        UpdateCovarianceMatrix();
    }

    virtual ~MVNLogisticPolicy()
    {

    }

    virtual inline std::string getPolicyName()
    {
        return "MVNLogisticPolicy";
    }
    virtual inline std::string getPolicyHyperparameters()
    {
        return "";
    }
    virtual inline std::string printPolicy()
    {
        return "";
    }

    virtual MVNLogisticPolicy* clone()
    {
        return new  MVNLogisticPolicy(*this);
    }

    // ParametricPolicy interface
public:
    virtual inline arma::vec getParameters() const
    {
        return arma::join_vert(approximator.getParameters(), mLogisticParams);
    }
    virtual inline const unsigned int getParametersSize() const
    {
        return approximator.getParametersSize() + mLogisticParams.n_elem;
    }
    virtual inline void setParameters(arma::vec &w)
    {
        assert(w.n_elem == this->getParametersSize());
        int dp = approximator.getParametersSize();
        arma::vec tmp = w.rows(0, dp-1);
        approximator.setParameters(tmp);
        for (int i = 0, ie = mLogisticParams.n_elem; i < ie; ++i)
        {
            mLogisticParams(i) = w[dp + i];
            assert(!std::isnan(mLogisticParams(i)) && !std::isinf(mLogisticParams(i)));
        }
        UpdateCovarianceMatrix();
    }

    // DifferentiablePolicy interface
public:

    virtual arma::vec difflog(const arma::vec& state, const arma::vec& action);

    virtual arma::mat diff2log(const arma::vec& state, const arma::vec& action);

private:

    /**
     * @brief The logistic function
     * @param w The exponent value
     * @param asymptote The asymptotic value
     * @return The value of the logistic function
     */
    inline double logistic(double w, double asymptote)
    {
        return asymptote / (1.0 + exp(-w));
    }

protected:

    /**
     * Compute the covariance matrix from the logistic parameters and
     * compute the Cholesky decomposition of it.
     *
     * @brief Update the covariance matrix
     */
    void UpdateCovarianceMatrix();


};

} // end namespace ReLe
#endif // NORMALPOLICY_H
