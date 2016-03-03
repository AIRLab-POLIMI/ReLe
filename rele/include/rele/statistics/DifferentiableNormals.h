#ifndef DIFFERENTIABLENORMALS_H_
#define DIFFERENTIABLENORMALS_H_

#include "rele/statistics/Distribution.h"

namespace ReLe
{


/**
 * @brief Parametric Normal distribution
 */
class ParametricNormal : public DifferentiableDistribution
{
public:

    ParametricNormal(unsigned int support_dim);

    ParametricNormal(const arma::vec& params,
                     const arma::mat& covariance);

    virtual ~ParametricNormal()
    { }

    // Distribution interface
public:
    virtual arma::vec operator() () override;
    virtual double operator() (arma::vec& point) override;

    virtual inline std::string getDistributionName() override
    {
        return "ParametricNormal";
    }

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) override;

    // DifferentiableDistribution interface
public:

    inline unsigned int getParametersSize() override
    {
        return mean.n_elem;
    }

    inline virtual arma::vec getParameters() override
    {
        return mean;
    }

    inline virtual void setParameters(arma::vec& newval)
    {
        mean = newval;
    }

    virtual void update(arma::vec &increment) override;
    virtual arma::vec difflog(const arma::vec &point) override;
    virtual arma::mat diff2log(const arma::vec &point) override;

public:
    virtual void writeOnStream(std::ostream &out);
    virtual void readFromStream(std::istream &in);


    // Specific Normal policy interface //TODO check this!!!
public:
    inline arma::vec getMean() const
    {
        return mean;
    }

    inline arma::mat getCovariance() const
    {
        return Cov;
    }

protected:
    /**
     * Compute mean, covariance, inverce covariance and determinat values
     * according to current parameterization.
     *
     * @brief Update internal state
     */
    virtual void updateInternalState();

protected:
    arma::vec mean;
    arma::mat Cov, invCov, cholCov;
    double detValue;

};

/**
 * Gaussian with mean and diagonal covariance.
 * Both mean and variance are learned.
 */
class ParametricDiagonalNormal : public ParametricNormal, public FisherInterface
{
public:
    ParametricDiagonalNormal(const arma::vec& mean, const arma::vec& covariance);

    virtual ~ParametricDiagonalNormal()
    {}

    virtual inline std::string getDistributionName() override
    {
        return "ParametricDiagonalNormal";
    }

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) override;

    // DifferentiableDistribution interface
public:
    arma::vec difflog(const arma::vec& point) override;
    arma::mat diff2log(const arma::vec& point) override;

    arma::sp_mat FIM() override;
    arma::sp_mat inverseFIM() override;

    // WritableInterface interface
public:
    void writeOnStream(std::ostream &out) override;
    void readFromStream(std::istream &in) override;

public:
    unsigned int getParametersSize() override;
    virtual arma::vec getParameters() override;
    virtual void setParameters(arma::vec& newval) override;
    virtual void update(arma::vec &increment) override;


    // ParametricNormal interface
protected:
    void updateInternalState() override;

private:
    arma::vec diagStdDev;
};


/**
 * @brief Parametric normal distribution with logistic diagonal variance
 *
 * This class represents a parametric Gaussian distribution with parameters \f$\rho\f$:
 * \f[x \sim \mathcal{N}(\cdot|\rho).\f]
 * The parameter vector \f$\rho\f$ is then defined as follows:
 * \f[\rho = [M, \Omega]^{T}\f]
 * where \f$M=[\mu_1,\dots,\mu_n]\f$, \f$\Omega = [\omega_1, \dots,\omega_n]\f$ and
 * \f$ n \f$ is the support dimension. As a consequence, the parameter
 * dimension is \f$2\cdot n\f$.
 *
 * Given a parametrization \f$\rho\f$, the distribution is defined by the mean
 * vector \f$M\f$ and a covariance matrix \f$\Sigma\f$.
 * In order to reduce the number of parameters, we discard
 * the cross-correlation terms in the covariance matrix: \f$ \Sigma = diag(\sigma_1,\dots,\sigma_n)\f$.
 * Moreover, in order to prevent the variance from becoming negative we exploit
 * the parametrization presented by Kimura and Kobayashi (1998),
 * where \f$\sigma_i\f$ is represented by a logistic function parameterized by \f$\omega_i\f$:
 * \f[\sigma_i = \frac{\tau}{1+e^{-\omega_i}}.\f]
 */
class ParametricLogisticNormal : public ParametricNormal
{


public:
    ParametricLogisticNormal(unsigned int point_dim,
                             double variance_asymptote);

    ParametricLogisticNormal(const arma::vec& mean, const arma::vec& logWeights,
                             double variance_asymptote);

    ParametricLogisticNormal(const arma::vec& variance_asymptote);

    ParametricLogisticNormal(const arma::vec& mean, const arma::vec& logWeights,
                             const arma::vec& variance_asymptote);

    virtual ~ParametricLogisticNormal()
    {}

    virtual inline std::string getDistributionName() override
    {
        return "ParametricLogisticNormal";
    }

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) override;

    // DifferentiableDistribution interface
public:
    arma::vec difflog(const arma::vec& point) override;
    arma::mat diff2log(const arma::vec& point) override;

    // WritableInterface interface
public:
    void writeOnStream(std::ostream &out) override;
    void readFromStream(std::istream &in) override;

public:
    unsigned int getParametersSize() override;
    virtual arma::vec getParameters() override;
    virtual void setParameters(arma::vec& newval) override;
    virtual void update(arma::vec &increment) override;


    // ParametricNormal interface
protected:
    void updateInternalState() override;

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
    arma::vec asVariance; //asymptotic varianceprivate:
    arma::vec logisticWeights; //weights used for the logistic function


};

class ParametricCholeskyNormal : public ParametricNormal, public FisherInterface
{

public:
    ParametricCholeskyNormal(const arma::vec& initial_mean,
                             const arma::mat& initial_cholA);

    virtual ~ParametricCholeskyNormal()
    {}

    virtual inline std::string getDistributionName() override
    {
        return "ParametricCholeskyNormal";
    }

    // DifferentiableDistribution interface
public:
    arma::vec difflog(const arma::vec& point) override;
    arma::mat diff2log(const arma::vec& point) override;

    arma::sp_mat FIM() override;
    arma::sp_mat inverseFIM() override;

    inline arma::mat getCholeskyDec()
    {
        return cholCov;
    }

    inline void setCholeskyDec(arma::mat& A)
    {
        cholCov = A;
        updateInternalState();
    }

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) override;

    // WritableInterface interface
public:
    void writeOnStream(std::ostream &out) override;
    void readFromStream(std::istream &in) override;

public:
    unsigned int getParametersSize() override;
    virtual arma::vec getParameters() override;
    virtual void setParameters(arma::vec& newval) override;
    virtual void update(arma::vec &increment) override;


    // ParametricNormal interface
protected:
    void updateInternalState() override;

};

class ParametricFullNormal : public ParametricNormal, public FisherInterface
{

public:
    ParametricFullNormal(const arma::vec& initial_mean,
                         const arma::mat& initial_cov);

    virtual ~ParametricFullNormal()
    {}

    virtual inline std::string getDistributionName() override
    {
        return "ParametricFullNormal";
    }

    // DifferentiableDistribution interface
public:
    arma::vec difflog(const arma::vec& point) override;
    arma::mat diff2log(const arma::vec& point) override;

    arma::sp_mat FIM() override;
    arma::sp_mat inverseFIM() override;

    virtual void wmle(const arma::vec& weights, const arma::mat& samples) override;

    // WritableInterface interface
public:
    void writeOnStream(std::ostream &out) override;
    void readFromStream(std::istream &in) override;

public:
    unsigned int getParametersSize() override;
    virtual arma::vec getParameters() override;
    virtual void setParameters(arma::vec& newval) override;
    virtual void update(arma::vec &increment) override;


    // ParametricNormal interface
protected:
    void updateInternalState() override;

};





} //end namespace

#endif //DIFFERENTIABLENORMALS_H_
