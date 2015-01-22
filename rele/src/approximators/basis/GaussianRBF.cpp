#include "basis/GaussianRBF.h"

using namespace arma;

namespace ReLe
{

GaussianRbf::GaussianRbf(unsigned int dimension, float mean_vec[], float scale_factor)
    : mean(arma::zeros<arma::vec>(dimension)), scale(scale_factor)
{
    if (dimension != 0)
    {
        for (unsigned i = 0; i < dimension; ++i)
        {
            mean[i] = mean_vec[i];
        }
    }
}

GaussianRbf::~GaussianRbf()
{
}

double GaussianRbf::operator()(const vec& input)
{
    double normv = 0.0;
    unsigned int dim = mean.n_rows;
    for (unsigned i = 0; i < dim; ++i)
    {
        normv += (input[i] - mean[i]) * (input[i] - mean[i]);
    }
    double retv = - sqrt(normv) / scale;
    retv = exp(retv);
    return retv;
}

void GaussianRbf::WriteOnStream(std::ostream &out)
{
    unsigned int dim = mean.n_rows;
    out << "GaussianRbf " << dim << std::endl;
    for (unsigned int i = 0; i < dim; i++)
    {
        out << mean[i] << " ";
    }
    out << scale;
}

void GaussianRbf::ReadFromStream(std::istream &in)
{
    unsigned int dim = mean.n_rows;
    in >> dim;

    mean.zeros(dim);
    double value;
    for (unsigned int i = 0; i < dim; i++)
    {
        in >> value;
        mean[i] = value;
    }
    in >> value;
    scale = value;
}


}
