#ifndef ARMADILLOPDFS_H_
#define ARMADILLOPDFS_H_

#include <armadillo>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ReLe
{

inline double normpdf(const double x, const double mean, const double var)
{
    return exp(-1.0 * ((x - mean) * (x - mean) / (2.0 * var)))
           / sqrt(2.0 * M_PI * var);
}

/**
 * Calculates the multivariate Gaussian probability density function.
 *
 * Example use:
 * @code
 * extern arma::vec x, mean;
 * extern arma::mat cov;
 * ....
 * double f = phi(x, mean, cov);
 * @endcode
 *
 * @param x Observation.
 * @param mean Mean of multivariate Gaussian.
 * @param cov Covariance of multivariate Gaussian.
 * @return Probability of x being observed from the given multivariate Gaussian.
 */
inline double mvnpdf(const arma::vec& x,
                     const arma::vec& mean,
                     const arma::mat& cov)
{
    arma::vec diff = x - mean;

    arma::vec prod = arma::solve(cov, diff);

    double exponent = -0.5 * arma::dot(diff, prod);

    return pow(2 * M_PI, x.n_elem / -2.0) * exp(exponent) / sqrt(arma::det(cov));
}

inline double mvnpdfFast(const arma::vec& x,
                         const arma::vec& mean,
                         const arma::mat& inverse_cov,
                         const double& det)
{
    arma::vec diff = x - mean;

    double exponent = -0.5 * arma::dot(diff, inverse_cov*diff);

    return pow(2 * M_PI, x.n_elem / -2.0) * exp(exponent) / sqrt(det);
}

/**
 * Calculates the multivariate Gaussian probability density function and also
 * the gradients of the mean and variance w.r.t. the input value x.
 *
 * Example use:
 * @code
 * extern arma::vec x, mean, g_mean, g_cov;
 * std::vector<arma::mat> d_cov; // the dSigma
 * ....
 * double f = phi(x, mean, cov, d_cov, &g_mean, &g_cov);
 * @endcode
 */
inline double mvnpdf(const arma::vec& x,
                     const arma::vec& mean,
                     const arma::mat& cov,
                     arma::vec& g_mean,
                     arma::vec& g_cov)
{

    //compute distance from mean
    arma::vec diff = x - mean;

    arma::mat cinv = arma::inv(cov);

    // compute exponent
    double exponent = -0.5 * arma::dot(diff, cinv*diff);

    long double f = pow(2.0 * M_PI, x.n_elem / -2.0)
                    * exp(exponent) / sqrt(arma::det(cov));

    // Calculate the gradient w.r.t. the input value x; this is a (dim x 1) vector.
    arma::vec invDiff = cinv*diff;
    g_mean = f * invDiff;

    // Calculate the g_cov values; this is a (1 x (dim * (dim + 1) / 2)) vector.
    g_cov = arma::vectorise(f * (cinv * diff * diff.t() * cinv - cinv));

    return f;
}

/**
 * Calculates the multivariate Gaussian probability density function and also
 * the gradients of the mean and variance w.r.t. the input value x.
 *
 * Example use:
 * @code
 * extern arma::vec x, mean, g_mean, g_cov;
 * std::vector<arma::mat> d_cov; // the dSigma
 * ....
 * double f = phi(x, mean, cov, d_cov, &g_mean, &g_cov);
 * @endcode
 */
inline double mvnpdfFast(const arma::vec& x,
                         const arma::vec& mean,
                         const arma::mat& inverse_cov,
                         const double& det,
                         arma::vec& g_mean,
                         arma::vec& g_cov)
{

    //compute distance from mean
    arma::vec diff = x - mean;

    // compute exponent
    double exponent = -0.5 * arma::dot(diff, inverse_cov * diff);

    long double f = pow(2.0 * M_PI, x.n_elem / -2.0)
                    * exp(exponent) / sqrt(det);

    // Calculate the gradient w.r.t. the input value x; this is a (dim x 1) vector.
    arma::vec invDiff = inverse_cov * diff;
    g_mean = f * invDiff;

    // Calculate the g_cov values; this is a (1 x (dim * (dim + 1) / 2)) vector.
    g_cov = f * (inverse_cov * diff * diff.t() * inverse_cov - inverse_cov);

    return f;
}

/**
 * Calculates the multivariate Gaussian probability density function for each
 * data point (column) in the given matrix, with respect to the given mean and
 * variance.
 *
 * @param x List of observations.
 * @param mean Mean of multivariate Gaussian.
 * @param cov Covariance of multivariate Gaussian.
 * @param probabilities Output probabilities for each input observation.
 */
inline void mvnpdf(const arma::mat& x,
                   const arma::vec& mean,
                   const arma::mat& cov,
                   arma::vec& probabilities)
{
    // Column i of 'diffs' is the difference between x.col(i) and the mean.
    arma::mat diffs = x - (mean * arma::ones<arma::rowvec>(x.n_cols));

    // Now, we only want to calculate the diagonal elements of (diffs' * cov^-1 *
    // diffs).  We just don't need any of the other elements.  We can calculate
    // the right hand part of the equation (instead of the left side) so that
    // later we are referencing columns, not rows -- that is faster.
    arma::mat rhs = -0.5 * inv(cov) * diffs;
    arma::vec exponents(diffs.n_cols); // We will now fill this.
    for (size_t i = 0; i < diffs.n_cols; i++)
        exponents(i) = exp(accu(diffs.unsafe_col(i) % rhs.unsafe_col(i)));

    probabilities = pow(2 * M_PI, mean.n_elem / -2.0) *
                    pow(det(cov), -0.5) * exponents;
}

inline arma::mat mvnrand(int n, arma::vec& mu, arma::mat& sigma)
{
    int ncols = sigma.n_cols;
    arma::mat Y = arma::randn(n, ncols);
    return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}

inline arma::vec mvnrand(arma::vec& mu, arma::mat& sigma)
{
    int ncols = sigma.n_cols;
    arma::mat Y = arma::randn(1, ncols);
    arma::mat temp = Y * arma::chol(sigma);
    return mu + temp.t();
}

inline arma::vec mvnrandFast(arma::vec& mu, arma::mat& CholSigma)
{
    int ncols = mu.n_rows;
    arma::mat Y = arma::randn(1, ncols);
    arma::mat temp = Y * CholSigma;
    return mu + temp.t();
}

} //end namespace

#endif //ARMADILLOPDFS_H_
