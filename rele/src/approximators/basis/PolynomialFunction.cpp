#include "basis/PolynomialFunction.h"

namespace ReLe
{

PolynomialFunction::PolynomialFunction(std::vector<unsigned int> dimension, std::vector<unsigned int> degree)
    : dimension(dimension), degree(degree)
{
}

PolynomialFunction::PolynomialFunction(unsigned int _dimension, unsigned int _degree)
    : dimension(_dimension)
{
    for (unsigned i = 0; i < _dimension; ++i)
    {
        degree.push_back(_degree);
    }
}

PolynomialFunction::~PolynomialFunction()
{}

double PolynomialFunction::operator()(const DenseArray &input)
{
    float result = 1.0;
    unsigned int i, j;
    for (i = 0; i < dimension.size(); i++)
    {
        for (j = 0; j < degree[i]; j++)
        {
            result *= input[dimension[i]];
        }
    }
    return result;
}

void PolynomialFunction::WriteOnStream(std::ostream &out)
{
    out << "Polynomial " << dimension.size() << std::endl;
    for (unsigned int i = 0; i < dimension.size(); i++)
    {
        out << dimension[i] << " " << degree[i] << std::endl;
    }
}

void PolynomialFunction::ReadFromStream(std::istream &in)
{
    unsigned int size = 0;
    in >> size;
    dimension.clear();
    degree.clear();
    unsigned int dim, deg;
    for (unsigned int i = 0; i < size; i++)
    {
        in >> dim >> deg;
        dimension.push_back(dim);
        degree.push_back(deg);
    }
}

}//end namespace
