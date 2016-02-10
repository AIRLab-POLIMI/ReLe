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
