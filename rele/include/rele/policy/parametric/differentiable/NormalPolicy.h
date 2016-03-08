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

#include "rele/policy/Policy.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/statistics/ArmadilloPDFs.h"

#define NORMALP_NAME "NormalPolicy"

namespace ReLe
{

// ================================================
// NORMAL POLICY
// ------------------------------------------------

/*!
 * Univariate normal policy with fixed standard deviation and mean linearly
 * parametrized:
 * \f[
 * \pi(x,u) = \mathcal{N}(a|\mu = \phi(x)*T \theta, \sigma^2)
 * \f]
 * The parameters to be optimized are \f$\theta_i\f$. Note that this class
 * assumes scalar actions.
 */
class NormalPolicy: public DifferentiablePolicy<DenseAction, DenseState>
{
public:
    /*!
     * Create a Normal policy with the given parameters. The initial
     * weights \f$\theta\f$ are set to zero.
     * \param initialStddev standard deviation \f$\sigma\f$
     * \param phi vector of basis functions \f$\phi\f$
     */
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

    /*!
     * Compute the mean given a state.
     * This function is invoked by the operators operator()(state,action)
     * and operator()(state) and by any other function that requires updated
     * information (e.g., difflog(state,action))
     * \param state the state to be evaluated.
     */
    virtual void calculateMeanAndStddev(const arma::vec& state);

public:

    virtual double operator()(const arma::vec& state, const arma::vec& action) override;

    virtual arma::vec operator()(const arma::vec& state) override;


    virtual inline std::string getPolicyName() override
    {
        return "NormalPolicy";
    }
    virtual inline std::string printPolicy() override
    {
        return "";
    }

    virtual NormalPolicy* clone() override
    {
        return new  NormalPolicy(*this);
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
    virtual arma::vec diff(const arma::vec& state, const arma::vec& action) override;

    virtual arma::vec difflog(const arma::vec& state, const arma::vec& action) override;

    virtual arma::mat diff2log(const arma::vec& state, const arma::vec& action) override;



protected:
    //mMean is used to store the mean value of a state
    double mInitialStddev, mMean;
    LinearApproximator approximator;
};

// ////////////////////////////////////////////////////////////////////////////////////
//  NORMAL POLICY WITH STATE DEPENDANT STDDEV (STD is not a parameter to be learned)
// ////////////////////////////////////////////////////////////////////////////////////

/**
 * Univariate normal policy with state dependant standard deviation
 * \f[
 * \pi(x,u) = \mathcal{N}(u| \mu=\phi(x)^T \omega, \sigma= \eta(x)^T k),
 * \f]
 * where \f$\theta\f$ are the parameters to be learned, \f$k\f$ is fixed.
 * Note that this class assumes scalar actions.
 */
class NormalStateDependantStddevPolicy: public NormalPolicy
{

public:
    /*!
     * Construct a normal policy and initialize the parameters \f$\theta\f$ to zero
     * \param phi the basis functions \f$\phi\f$ of the mean
     * \param stdPhi the basis functions \f$\eta\f$ of the standard deviation
     * \param stdDevParameters the weights \f$k\f$ of the linear approximator of the standard deviation
     */
    NormalStateDependantStddevPolicy(Features& phi,
                                     Features& stdPhi, arma::vec& stdDevParameters) :
        NormalPolicy(1, phi), stdApproximator(stdPhi)
    {
        stdApproximator.setParameters(stdDevParameters);
    }

    virtual ~NormalStateDependantStddevPolicy()
    {

    }


    virtual inline std::string getPolicyName() override
    {
        return "NormalStateDependantStddevPolicy";
    }

    virtual inline std::string printPolicy() override
    {
        return "";
    }

    virtual NormalStateDependantStddevPolicy* clone() override
    {
        return new  NormalStateDependantStddevPolicy(*this);
    }

protected:

    NormalStateDependantStddevPolicy(Features& phi,
                                     Features& stdPhi)
        : NormalPolicy(1, phi), stdApproximator(stdPhi)
    {
    }

    /*!
     * Compute the mean and standard deviation given a state
     * \param state the state to be evaluated
     */
    virtual void calculateMeanAndStddev(const arma::vec& state) override;

protected:
    LinearApproximator stdApproximator;

};


// ////////////////////////////////////////////////////////////////////////////////////
// NORMAL POLICY WITH LEARNED STATE DEPENDANT STDDEV (parameters: mean and standard deviations)
// ////////////////////////////////////////////////////////////////////////////////////

/**
 * Univariate normal policy with state dependant standard deviation
 * \f[
 * \pi(x,u) = \mathcal{N}(u| \mu=\phi(x)^T \omega, \sigma= \eta(x)^T k),
 * \f]
 * where \f$\theta = [\omega, k]\f$ are the parameters to be learned.
 * Note that this class assumes scalar actions.
 */
class NormalLearnableStateDependantStddevPolicy : public NormalStateDependantStddevPolicy
{
public:
    /*!
     * Construct a Normal policy and initialize the parameters \f$\theta\f$ to zero.
     * \param phi the basis functions \f$\phi\f$ of the mean
     * \param stdPhi the basis functions \f$\eta\f$ of the standard deviation
     */
    NormalLearnableStateDependantStddevPolicy(Features& phi, Features& stdPhi) :
        NormalStateDependantStddevPolicy(phi,stdPhi)
    {
        arma::vec w(getParametersSize(), arma::fill::ones);
        setParameters(w);
    }

    /*!
     * Construct a Normal policy with the provided weights.
     * \param phi the basis functions \f$\phi\f$ of the mean
     * \param stdPhi the basis functions \f$\eta\f$ of the standard deviation
     * \param w the parameters \f$\theta=[\omega,k]\f$ to be set
     */
    NormalLearnableStateDependantStddevPolicy(Features& phi, Features& stdPhi,
            arma::vec& w) :
        NormalStateDependantStddevPolicy(phi,stdPhi)
    {
        setParameters(w);
    }

    virtual ~NormalLearnableStateDependantStddevPolicy()
    {
    }

    virtual inline std::string getPolicyName() override
    {
        return "NormalLearnableStddevPolicy";
    }

    virtual NormalLearnableStateDependantStddevPolicy* clone() override
    {
        return new  NormalLearnableStateDependantStddevPolicy(*this);
    }

    // ParametricPolicy interface
public:
    /*!
     * Return the parameters \f$\theta\f$ as a concatenation of \f$[\omega,k\f$.
     * \return the parameters \f$\theta\f$
     */
    virtual inline arma::vec getParameters() const override
    {
        return vectorize(approximator.getParameters(),stdApproximator.getParameters());
    }
    virtual inline const unsigned int getParametersSize() const override
    {
        return approximator.getParametersSize() + stdApproximator.getParametersSize();
    }
    /*!
     * Set the policy parameters. \f$w\f$ must have dimension \f$|\omega| + |k|\f$.
     * The vector \f$w\f$ is obtained as \f$[\omega,k]\f$.
     * \param w the new parameters
     */
    virtual inline void setParameters(const arma::vec& w) override
    {
        int n = getParametersSize();
        assert(w.size() == n);

        int nbMeanP = approximator.getParametersSize();
        approximator.setParameters(w.rows(0,nbMeanP-1));
        stdApproximator.setParameters(w.rows(nbMeanP, n-1));
    }


    // DifferentiablePolicy interface
public:
    virtual arma::vec difflog(const arma::vec& state, const arma::vec& action) override;

    virtual arma::mat diff2log(const arma::vec& state, const arma::vec& action) override;

};


///////////////////////////////////////////////////////////////////////////////////////
/// MVN POLICY
///////////////////////////////////////////////////////////////////////////////////////

/*!
 * This class represents a multivariate Normal policy with fixed covariance matrix
 * and linear approximation of the mean value:
 * \f[ \pi (x,u) = \mathcal{N}(a| \mu=\phi(x)^T\theta, \Sigma),\qquad
 * \forall s \in R^{n_x}, a \in R^{n_u},\f]
 * where \f$\phi(x)\f$ is a \f$(k \times n_x)\f$ matrix and
 * \f$\theta\f$ is a \f$k\f$-dimensional vector. The parameters to be learned
 * are \f$\theta\f$.
 */
class MVNPolicy: public DifferentiablePolicy<DenseAction, DenseState>
{
public:

    /*!
     * Create an instance of the class using the given basis functions.
     * Covariance matrix is initialized to the identity matrix.
     * Note that the weights of the mean approximation are set to zero.
     *
     * \param phi the basis functions
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

    /*!
     * Create an instance of the class using the given basis functions
     * and covariance matrix.
     * Note that the weights of the mean approximation are set to zero.
     *
     * \param phi the basis functions
     * \param covariance the covariance matrix (\f$n_u \times n_u\f$)
     */
    MVNPolicy(Features& phi, arma::mat& covariance) :
        approximator(phi),
        mMean(approximator.getOutputSize(), arma::fill::zeros),
        mCovariance(covariance)
    {
        mCinv = arma::inv(mCovariance);
        mCholeskyDec = arma::chol(mCovariance);
        mDeterminant = arma::det(mCovariance);
    }

    /*!
     * Create an instance of the class using the given basis functions
     * and the covariance matrix.
     *
     * \param projector the basis functions
     * \param initialCov The covariance matrix (\f$n_u \times n_u\f$) as list
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

    /*!
     * Create an instance of the class using the given basis functions
     * and the covariance matrix.
     *
     * \param projector the basis functions
     * \param initialCov The covariance matrix (\f$n_u \times n_u\f$) as an array
     */
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

    virtual inline std::string getPolicyName() override
    {
        return "MVNPolicy";
    }

    virtual inline std::string printPolicy() override
    {
        return "";
    }

public:

    virtual double operator()(const arma::vec& state, const arma::vec& action) override;

    virtual arma::vec operator()(const arma::vec& state) override;

    virtual MVNPolicy* clone() override
    {
        return new  MVNPolicy(*this);
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
    virtual arma::vec diff(const arma::vec& state, const arma::vec& action) override;

    virtual arma::vec difflog(const arma::vec& state, const arma::vec& action) override;

    virtual arma::mat diff2log(const arma::vec& state, const arma::vec& action) override;

protected:

    /*!
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
     * \param state The state where the policy is evaluated.
     * \param cholesky_dec A flag used to require the Cholesky decomposition of the
     * covariance matrix. By default is false.
     */
    inline virtual void updateInternalState(const arma::vec& state, bool cholesky_dec = false)
    {
        // compute mean vector
        mMean = approximator(state);

    }

protected:
    arma::mat mCovariance, mCinv, mCholeskyDec;
    double mDeterminant;
    LinearApproximator approximator;
    arma::vec mMean;
};


// ////////////////////////////////////////////////////////////////////////////////////
// MVN POLICY with state dependant covariance
// ////////////////////////////////////////////////////////////////////////////////////
/**
 * Multivariate Normal policy with state dependant covariance matrix
 * \f[
 * \pi(x,u) = \mathcal{N}(u| \mu=\phi(x)^T \theta, \sigma= K \eta(x)),
 * \f]
 * where \f$\theta\f$ are the parameters to be learned, \f$K\f$ is fixed.
 */
class MVNStateDependantStddevPolicy : public MVNPolicy
{
public:
    /*!
     * Create an instance of the class using the given basis functions and
     * the covariance parameters
     * \param phi the basis function of the mean
     * \param phiCov the basis function of the covariance matrix
     * \param CovW the parameters of the covariance approximation
     */
    MVNStateDependantStddevPolicy(Features& phi, Features& phiCov, arma::mat& CovW)
        : MVNPolicy(phi), phiStdDev(phiCov), stdDevW(CovW)
    {
        assert(phiCov.cols() == phi.cols());
        assert(CovW.n_rows == phiCov.cols());
        assert(CovW.n_cols == phiCov.rows());
    }

    virtual inline std::string getPolicyName() override
    {
        return "MVNStateDependantStddevPolicy";
    }

    virtual inline std::string printPolicy() override
    {
        return "";
    }

protected:
    /*!
     * Update both mean and covariance
     * \see{MVNPolicy::updateInternalState}
     */
    inline virtual void updateInternalState(const arma::vec& state, bool cholesky_dec = false) override
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

    virtual inline std::string getPolicyName() override
    {
        return "MVNDiagonalPolicy";
    }

    virtual inline std::string printPolicy() override
    {
        return "";
    }

    virtual MVNDiagonalPolicy* clone() override
    {
        return new  MVNDiagonalPolicy(*this);
    }

    // ParametricPolicy interface
public:
    virtual inline arma::vec getParameters() const override
    {
        return arma::join_vert(approximator.getParameters(), stddevParams);
    }
    virtual inline const unsigned int getParametersSize() const override
    {
        return approximator.getParametersSize() + stddevParams.n_elem;
    }
    virtual void setParameters(const arma::vec& w) override;

    // DifferentiablePolicy interface
public:

    virtual arma::vec difflog(const arma::vec& state, const arma::vec& action) override;

    virtual arma::mat diff2log(const arma::vec& state, const arma::vec& action) override;

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

    virtual inline std::string getPolicyName() override
    {
        return "MVNLogisticPolicy";
    }

    virtual inline std::string printPolicy() override
    {
        return "";
    }

    virtual MVNLogisticPolicy* clone() override
    {
        return new  MVNLogisticPolicy(*this);
    }

    // ParametricPolicy interface
public:
    virtual inline arma::vec getParameters() const override
    {
        return arma::join_vert(approximator.getParameters(), mLogisticParams);
    }
    virtual inline const unsigned int getParametersSize() const override
    {
        return approximator.getParametersSize() + mLogisticParams.n_elem;
    }
    virtual inline void setParameters(const arma::vec& w) override
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

    virtual arma::vec difflog(const arma::vec& state, const arma::vec& action) override;

    virtual arma::mat diff2log(const arma::vec& state, const arma::vec& action) override;

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
