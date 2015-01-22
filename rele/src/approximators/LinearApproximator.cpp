#include "LinearApproximator.h"
#include <cassert>

using namespace std;
using namespace arma;

namespace ReLe
{

LinearApproximator::LinearApproximator(unsigned int input_dim, unsigned int output_dim)
    : ParametricRegressor(input_dim, output_dim)
{
    assert(output_dim == 1);
}

LinearApproximator::LinearApproximator(const unsigned int input_dim, BasisFunctions& bfs)
    : ParametricRegressor(input_dim, 1), basis(bfs),
      parameters(bfs.size(), fill::zeros)
{
}

LinearApproximator::~LinearApproximator()
{
}

void LinearApproximator::evaluate(const vec& input, vec& output)
{
    output[0] = basis.dot(input, parameters);
}

}
