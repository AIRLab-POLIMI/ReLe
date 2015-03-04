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

void GaussianRbf::writeOnStream(std::ostream &out)
{
    unsigned int dim = mean.n_rows;
    out << "GaussianRbf " << dim << std::endl;
    for (unsigned int i = 0; i < dim; i++)
    {
        out << mean[i] << " ";
    }
    out << scale;
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
    in >> value;
    scale = value;
}


}
