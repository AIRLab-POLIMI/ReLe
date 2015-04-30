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

#ifndef INCLUDE_RELE_APPROXIMATORS_FEATURES_DENSEFEATURES_H_
#define INCLUDE_RELE_APPROXIMATORS_FEATURES_DENSEFEATURES_H_

#include "Features.h"
#include <cassert>

namespace ReLe
{

template<class InputC>
class DenseFeatures_: public Features_<InputC>
{

public:
    DenseFeatures_(BasisFunction_<InputC>* basisFunction) : basis(1)
    {
        basis[0] = basisFunction;
    }

    DenseFeatures_(BasisFunctions_<InputC>& basisVector) :  basis(basisVector.size())
    {

        for(unsigned int i = 0; i < basisVector.size(); i++)
        {
            basis[i] = basisVector[i];
        }
    }

    DenseFeatures_(BasisFunctions_<InputC>& basisVector, unsigned int rows, unsigned int cols)
        : basis(rows, cols)
    {
        assert(rows*cols == basisVector.size());

        for(unsigned int i = 0; i < basisVector.size(); i++)
        {
            basis[i] = basisVector[i];
        }
    }

    virtual ~DenseFeatures_()
    {
        for(auto bf : basis)
        {
            delete bf;
        }
    }

    virtual arma::mat operator()(const InputC& input)
    {
        arma::mat output(basis.n_rows, basis.n_cols);

        for(unsigned int i = 0; i < basis.n_elem; i++)
        {
            BasisFunction_<InputC>& bf = *basis[i];
            output[i] = bf(input);
        }

        return output;
    }

    inline virtual size_t rows() const
    {
        return basis.n_rows;
    }

    inline virtual size_t cols() const
    {
        return basis.n_cols;
    }

private:
    arma::field<BasisFunction_<InputC>*> basis;

};

typedef DenseFeatures_<arma::vec> DenseFeatures;


}

#endif /* INCLUDE_RELE_APPROXIMATORS_FEATURES_DENSEFEATURES_H_ */
