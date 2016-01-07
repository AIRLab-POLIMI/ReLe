#ifndef GAUSSIANRBF_H
#define GAUSSIANRBF_H

#include "rele/approximators/BasisFunctions.h"
#include <armadillo>

namespace ReLe
{

class GaussianRbf : public BasisFunction
{

public:

    GaussianRbf(double center, double width, bool useSquareRoot = false);
    GaussianRbf(arma::vec center, double width, bool useSquareRoot = false);
    GaussianRbf(arma::vec center, arma::vec width, bool useSquareRoot = false);

    virtual ~GaussianRbf();
    double operator() (const arma::vec& input) override;

    inline arma::vec& getCenter()
    {
        return mean;
    }

    inline arma::vec& getWidth()
    {
        return scale;
    }

    inline unsigned int getSize()
    {
        return mean.n_rows;
    }

    /**
     * @brief generate
     * It computes equally distributed radial basis functions with 25% of
     * overlapping and confidence between 95-99%.
     *
     * @param n_centers number of centers (same for all dimensions)
     * @param range N-by-2 matrix with min and max values for the N-dimensional input state
     * @return the set of Gaussin RBF
     */
    static BasisFunctions generate(arma::vec& numb_centers, arma::mat& range);

    static BasisFunctions generate(unsigned int n_centers, std::initializer_list<double> l);
    static BasisFunctions generate(std::initializer_list<unsigned int> n_centers, std::initializer_list<double> l);
    static BasisFunctions generate(arma::mat& centers, arma::mat& widths);


    virtual void writeOnStream (std::ostream& out) override;
    virtual void readFromStream(std::istream& in) override;

private:
    arma::vec mean, scale;
    bool squareRoot;
};

}//end namespace

#endif // GAUSSIANRBF_H
