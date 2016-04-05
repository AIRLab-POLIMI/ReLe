#ifndef GAUSSIANRBF_H
#define GAUSSIANRBF_H

#include "rele/approximators/BasisFunctions.h"
#include <armadillo>

namespace ReLe
{

/*!
 * This class implements Gaussian Radial Basis Functions.
 */
class GaussianRbf : public BasisFunction
{

public:
    /*!
     * Constructor.
     * \param center the center of the Gaussian radial function
     * \param width the width of the Gaussian radial function
     * \param useSquareRoot specify whether to use square root of the exponential term
     * of the Gaussian radial function
     */
    GaussianRbf(double center, double width, bool useSquareRoot = false);

    /*!
     * Constructor.
     * \param center vector of centers of the Gaussian radial function
     * \param width the width of the Gaussian radial function
     * \param useSquareRoot specify whether to use square root of the exponential term
     * of the Gaussian radial function
     */
    GaussianRbf(arma::vec center, double width, bool useSquareRoot = false);

    /*!
     * Constructor.
     * \param center vector of centers of the Gaussian radial function
     * \param width vector of widths of the Gaussian radial function
     * \param useSquareRoot specify whether to use square root of the exponential term
     * of the Gaussian radial function
     */
    GaussianRbf(arma::vec center, arma::vec width, bool useSquareRoot = false);

    /*!
     * Destructor.
     */
    virtual ~GaussianRbf();

    double operator() (const arma::vec& input) override;

    /*!
     * Getter.
     * \return center of the Gaussian radial function
     */
    inline arma::vec& getCenter()
    {
        return mean;
    }

    /*!
     * Getter.
     * \return width of the Gaussian radial function
     */
    inline arma::vec& getWidth()
    {
        return scale;
    }

    /*!
     * Getter.
     * \return the number of centers
     */
    inline unsigned int getSize()
    {
        return mean.n_rows;
    }

    /*!
     * Return the Gaussian Radial Basis Functions with given number of centers and intervals.
     * \param numb_centers vector of number of centers
     * \param range interval matrix
     * \return the generated basis functions
     */
    static BasisFunctions generate(arma::vec& numb_centers, arma::mat& range);

    /*!
     * Return the Gaussian Radial Basis Functions with given number of centers and intervals.
     * \param n_centers number of centers
     * \param l list of intervals
     * \return the generated basis functions
     */
    static BasisFunctions generate(unsigned int n_centers, std::initializer_list<double> l);

    /*!
     * Return the Gaussian Radial Basis Functions with given number of centers and intervals.
     * \param n_centers list of number of centers
     * \param l list of intervals
     * \return the generated basis functions
     */
    static BasisFunctions generate(std::initializer_list<unsigned int> n_centers, std::initializer_list<double> l);

    /*!
     * Return the Gaussian Radial Basis Functions with given centers and widths.
     * \param centers vector of centers of the Gaussian radial functions
     * \param widths vector of widths of the Gaussian radial functions
     * \return the generated basis functions
     */
    static BasisFunctions generate(arma::mat& centers, arma::mat& widths);

    virtual void writeOnStream (std::ostream& out) override;
    virtual void readFromStream(std::istream& in) override;

private:
    arma::vec mean, scale;
    bool squareRoot;
};

}//end namespace

#endif // GAUSSIANRBF_H
