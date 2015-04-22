/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef INCLUDE_RELE_APPROXIMATORS_FEATURES_DENSEFEATURES_H_
#define INCLUDE_RELE_APPROXIMATORS_FEATURES_DENSEFEATURES_H_

#include "Features.h"

namespace ReLe
{


class DenseFeatures: public Features
{
public:
    DenseFeatures(BasisFunction* basisVector);
    DenseFeatures(BasisFunctions& basisVector);
    DenseFeatures(BasisFunctions& basisVector, unsigned int rows, unsigned int cols);
    virtual ~DenseFeatures();
    virtual arma::mat operator()(const arma::vec& input);

    inline virtual size_t rows() const
    {
        return basis.n_rows;
    }

    inline virtual size_t cols() const
    {
        return basis.n_cols;
    }

private:
    arma::field<BasisFunction*> basis;

};

}

#endif /* INCLUDE_RELE_APPROXIMATORS_FEATURES_DENSEFEATURES_H_ */
