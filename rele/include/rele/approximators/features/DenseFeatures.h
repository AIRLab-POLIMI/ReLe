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

#include "rele/approximators/Features.h"
#include <cassert>

namespace ReLe
{

/*!
 * This class implements a dense features matrix.
 * A dense features matrix is a feature matrix where all the elements
 * are specified as a set of basis functions.
 * The evaluation of this features class returns a dense matrix.
 */
template<class InputC>
class DenseFeatures_: public Features_<InputC>
{

public:
    /*!
     * Constructor.
     * Construct a single feature matrix (a scalar).
     * \param basisFunction the basis function rappresenting this feature.
     */
    DenseFeatures_(BasisFunction_<InputC>* basisFunction) : basis(1)
    {
        basis[0] = basisFunction;
    }

    /*!
     * Constructor.
     * Construct a feature vector.
     * \param basisVector the set of basis functions to use
     */
    DenseFeatures_(BasisFunctions_<InputC>& basisVector) :  basis(basisVector.size())
    {

        for(unsigned int i = 0; i < basisVector.size(); i++)
        {
            basis[i] = basisVector[i];
        }
    }

    /*!
     * Constructor.
     * Construct a feature matrix.
     * \param basisVector the set of basis functions to use
     * \param rows the number of rows of the feature matrix
     * \param cols the number of cols of the feature matrix
     */
    DenseFeatures_(BasisFunctions_<InputC>& basisVector, unsigned int rows, unsigned int cols)
        : basis(rows, cols)
    {
        assert(rows*cols == basisVector.size());

        for(unsigned int i = 0; i < basisVector.size(); i++)
        {
            basis[i] = basisVector[i];
        }
    }

    /*!
     * Destructor.
     * Destroys also all the given basis.
     */
    virtual ~DenseFeatures_()
    {
        for(auto bf : basis)
        {
            delete bf;
        }
    }

    virtual arma::mat operator()(const InputC& input) override
    {
        arma::mat output(basis.n_rows, basis.n_cols);

        for(unsigned int i = 0; i < basis.n_elem; i++)
        {
            BasisFunction_<InputC>& bf = *basis[i];
            output[i] = bf(input);
        }

        return output;
    }

    inline virtual size_t rows() const override
    {
        return basis.n_rows;
    }

    inline virtual size_t cols() const override
    {
        return basis.n_cols;
    }

private:
    arma::field<BasisFunction_<InputC>*> basis;

};

//! Template alias.
typedef DenseFeatures_<arma::vec> DenseFeatures;


}

#endif /* INCLUDE_RELE_APPROXIMATORS_FEATURES_DENSEFEATURES_H_ */
