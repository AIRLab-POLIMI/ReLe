/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
 * Versione 1.0
 *
 * This file is part of rele.
 *
 * rele is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rele is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with rele.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "rele/approximators/basis/GaussianRbf.h"
#include <cassert>

using namespace arma;

namespace ReLe
{
GaussianRbf::GaussianRbf(double center, double width, bool useSquareRoot)
    : mean(arma::ones<arma::vec>(1)*center),
      scale(arma::ones<arma::vec>(1)*width), squareRoot(useSquareRoot)
{
}

GaussianRbf::GaussianRbf(arma::vec center, double width, bool useSquareRoot)
    : mean(center), scale(arma::ones<arma::vec>(center.n_elem)*width), squareRoot(useSquareRoot)
{

}

GaussianRbf::GaussianRbf(arma::vec center, arma::vec width, bool useSquareRoot)
    : mean(center), scale(width), squareRoot(useSquareRoot)
{
    assert(mean.n_elem == scale.n_elem);
}


GaussianRbf::~GaussianRbf()
{
}

double GaussianRbf::operator()(const vec& input)
{
    double retv = 0.0;
    unsigned int dim = mean.n_rows;

    if (squareRoot)
    {
        //TODO [IMPORTANT] REMOVE usato per la diga
        for (unsigned i = 0; i < dim; ++i)
        {
            retv += (input[i] - mean[i]) * (input[i] - mean[i]) / (scale(i)*scale(i));
        }
        retv = sqrt(retv);
    }
    else
    {
        for (unsigned i = 0; i < dim; ++i)
        {
            retv += (input[i] - mean[i]) * (input[i] - mean[i]) / scale(i);
        }
    }
    retv = exp(-retv);
    return retv;
}

void uniform_grid(arma::mat& grid, int& nrows, int& ncols, arma::vec& discrete_values)
{

    int i1 = 0, dim = discrete_values.n_elem;

    for (int i = 0; i < dim; ++i)
    {

        // replicate grid for each term i1
        for (int r = 0; r < nrows; ++r)
        {
            int idxr = r + i * nrows;
            for(int c = 0; c < ncols; ++c)
            {
                grid(idxr,c) = grid(r,c);
            }
            grid(idxr,ncols) = discrete_values[i1];
        }

        //next cordinate
        i1++;
    }

    ncols++;
    nrows *= dim;

}

BasisFunctions GaussianRbf::generate(arma::vec& numb_centers, arma::mat& range)
{
    assert(numb_centers.n_elem == range.n_rows);
    assert(2 == range.n_cols);

    BasisFunctions basis;

    int n_features = range.n_rows;
    arma::vec b(n_features, arma::fill::zeros);
    std::vector<arma::vec> c(n_features, arma::vec());

    // compute bandwidths and centers for each dimension
    int totpoints = 1;
    for (int i = 0; i < n_features; ++i)
    {
        int n_centers = numb_centers[i];
        b(i) = (range(i,1) - range(i,0))*(range(i,1) - range(i,0)) / (n_centers*n_centers*n_centers);
        double m = fabs(range(i,1) - range(i,0)) / n_centers;
        if (n_centers == 1)
        {
            // when the number of centers is 1, linspace returns the upper bound
            // we want the average value
            c[i] = (range(i,1) + range(i,0))/2.0;
        }
        else
        {

            c[i] = arma::linspace<arma::vec>(-m * 0.1 + range(i,0), range(i,1) + m * 0.1, n_centers);
        }
        totpoints *= n_centers;
    }

    int grid_nrows = 1;
    int grid_ncols = 0;


    //allocate space for the grid
    arma::mat grid(totpoints, n_features);

    // fill the grid with the first points
    for (int i = 0; i < n_features; ++i)
    {
        uniform_grid(grid, grid_nrows, grid_ncols, c[i]);
    }

    //    std::cerr << std::endl << grid <<std::endl;
    //    std::cerr << std::endl << b.t() <<std::endl;
    for (int i=0; i< totpoints; ++i)
    {
        arma::mat v = grid.row(i);
        GaussianRbf* bf = new GaussianRbf(v.t(), b, false);
        basis.push_back(bf);
    }
    return basis;
}

BasisFunctions GaussianRbf::generate(unsigned int n_centers, std::initializer_list<double> l)
{
    int dim = l.size();
    assert(dim % 2 == 0);
    arma::mat range(dim/2, 2);
    int row = 0, col = 0;
    for (auto v : l)
    {
        range(row, col++) = v;
        if (col == 2)
        {
            row++;
            col = 0;
        }
    }
    arma::vec centerDim(dim/2);
    centerDim.fill(n_centers);
    return GaussianRbf::generate(centerDim, range);
}

BasisFunctions GaussianRbf::generate(std::initializer_list<unsigned int> n_centers, std::initializer_list<double> l)
{
    int dim = l.size();
    assert(dim % 2 == 0);
    assert(dim / 2 == n_centers.size());
    arma::mat range(dim/2, 2);
    int row = 0, col = 0;
    for (auto v : l)
    {
        range(row, col++) = v;
        if (col == 2)
        {
            row++;
            col = 0;
        }
    }
    arma::vec centerDim(dim/2);
    int i = 0;
    for (auto v : n_centers)
    {
        centerDim[i++] = v;
    }
    return GaussianRbf::generate(centerDim, range);
}

BasisFunctions GaussianRbf::generate(arma::mat& centers, arma::mat& widths)
{
    assert(centers.n_rows == widths.n_rows);
    assert(centers.n_cols == widths.n_cols);
    int ncols = centers.n_cols;
    BasisFunctions basis;
    for (int i = 0; i < ncols; i++)
    {
        basis.push_back(new GaussianRbf(centers.col(i), widths.col(i)));
    }
    return basis;
}

void GaussianRbf::writeOnStream(std::ostream &out)
{
    unsigned int dim = mean.n_rows;
    out << "GaussianRbf " << dim << std::endl;
    for (unsigned int i = 0; i < dim; i++)
    {
        out << mean[i] << " ";
    }
    out << endl;
    for (unsigned int i = 0; i < dim; i++)
        out << scale[i] << " ";
    out << endl;
    out << (squareRoot?1:0) << endl;
}

void GaussianRbf::readFromStream(std::istream &in)
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
    for (unsigned int i = 0; i < dim; i++)
    {
        in >> value;
        scale[i] = value;
    }
    in >> value;
    squareRoot = true;
    if (value == 0)
        squareRoot = false;
}


}
