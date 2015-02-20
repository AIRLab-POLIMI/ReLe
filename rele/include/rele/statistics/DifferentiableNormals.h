#ifndef DIFFERENTIABLENORMALS_H_
#define DIFFERENTIABLENORMALS_H_

#include "Distribution.h"

namespace ReLe
{


/**
 * @brief Parametric Normal distribution
 */
class ParametricNormal : public DifferentiableDistribution
{
public:

    ParametricNormal(unsigned int support_dim,
                     unsigned int param_size);

    ParametricNormal(unsigned int support_dim,
                     arma::vec& params,
                     arma::mat& covariance);

    virtual ~ParametricNormal()
    { }

    // Distribution interface
public:
    virtual arma::vec operator() ();
    virtual double operator() (arma::vec &point);

    // DifferentiableDistribution interface
public:
    virtual void Update(arma::vec &increment);
    virtual arma::vec difflog(const arma::vec &point);
    virtual arma::mat diff2Log(const arma::vec &point);

    // WritableInterface interface
public:
    virtual void WriteOnStream(std::ostream &out);
    virtual void ReadFromStream(std::istream &in);

protected:
    /**
     * Compute mean, covariance, inverce covariance and determinat values
     * according to current parameterization.
     *
     * @brief Update internal state
     */
    virtual void UpdateInternalState();

protected:
    arma::vec parameters, mean;
    arma::mat Cov, invCov;
    double detValue;

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

    ParametricLogisticNormal(unsigned int point_dim,
                             double variance_asymptote,
                             arma::vec& params);

    virtual ~ParametricLogisticNormal()
    {}

    // DifferentiableDistribution interface
public:
    arma::vec difflog(const arma::vec& point);
    arma::mat diff2Log(const arma::vec& point);

    // WritableInterface interface
public:
    void WriteOnStream(std::ostream &out);
    void ReadFromStream(std::istream &in);


    // ParametricNormal interface
protected:
    void UpdateInternalState();

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
    double asVariance; //asymptotic variance


};


} //end namespace

#endif //DIFFERENTIABLENORMALS_H_
