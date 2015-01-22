#ifndef GAUSSIANRBF_H
#define GAUSSIANRBF_H

#include "BasisFunctions.h"
#include <armadillo>

namespace ReLe
{

class GaussianRbf : public BasisFunction
{

public:

    GaussianRbf(unsigned int dimension = 0, float mean_vec[] = 0, float scale_factor = 0);
    virtual ~GaussianRbf();
    double operator() (const arma::vec& input);

    void GetMean(arma::vec& meanval)
    {
        meanval = mean;
    }

    unsigned int GetDimension()
    {
        return mean.n_rows;
    }


    virtual void WriteOnStream (std::ostream& out);
    virtual void ReadFromStream(std::istream& in);

private:
    arma::vec mean;
    double scale;
};

}//end namespace

#endif // GAUSSIANRBF_H
