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

#ifndef INCLUDE_RELE_APPROXIMATORS_FEATURES_SPARSEFEATURES_H_
#define INCLUDE_RELE_APPROXIMATORS_FEATURES_SPARSEFEATURES_H_

#include "rele/approximators/Features.h"

namespace ReLe
{

/*!
 * This class implement a sparse features matrix.
 * A sparse feature matrix is a matrix where some basis function
 * are by default zero valued matrix.
 * Despite the name, the evaluation of this features class returns a dense matrix.
 */
template<class InputC>
class SparseFeatures_: public Features_<InputC>
{

public:
    /*!
     * Constructor.
     * Construct an empty set of sparse feature.
     */
    SparseFeatures_(): n_rows(0), n_cols(0)
    {
    }

    /*!
     * Constructor.
     * Constructs a set of sparse features from the given basis functions.
     * \param basis the basis functions to use
     * \param replicationN the number of times the features should be replicated
     * \param indipendent if the features replicated should be padded by zeros,
     *  to avoid common parameters.
     */
    SparseFeatures_(BasisFunctions_<InputC>& basis,
                    unsigned int replicationN = 1,
                    bool indipendent = true)
        : n_rows(basis.size()*(indipendent?replicationN:1)), n_cols(replicationN), valuesToDelete(basis)
    {
        unsigned int i, k;
        unsigned int offset = 0;
        unsigned int nBasis = basis.size();
        for (k = 0; k < replicationN; ++k)
        {
            for (i = 0; i < nBasis; ++i)
            {
                //create triplet (row,col,val)
                rowsIdxs.push_back(offset+i);
                colsIdxs.push_back(k);
                values.push_back(basis[i]);
            }

            if (indipendent)
            {
                offset += nBasis;
            }
        }
    }

    virtual arma::mat operator ()(const InputC& input) override
    {
        unsigned int c, r, i, nelem = rowsIdxs.size();
        double val;
        arma::mat F(n_rows, n_cols, arma::fill::zeros);

        for (i = 0; i < nelem; ++i)
        {
            r = rowsIdxs[i];
            c = colsIdxs[i];
            BasisFunction& bfs = *(values[i]);
            val = bfs(input);

            F(r,c) = val;
        }

        return F;
    }

    virtual inline size_t rows() const override
    {
        return n_rows;
    }

    virtual inline size_t cols() const override
    {
        return n_cols;
    }

    /*!
     * Adds a single basis function
     * \param row the row where to add the basis function
     * \param col the column where to add the basis function
     * \param bfs the basis function to add
     */
    void addBasis(unsigned int row, unsigned int col, BasisFunction_<InputC>* bfs)
    {
        unsigned int c, r, i, nelem = rowsIdxs.size();
        bool found = false;
        for (i = 0; i < nelem && !found; ++i)
        {
            r = rowsIdxs[i];
            c = colsIdxs[i];
            if ((r == row) && (c == col))
            {
                found = true;
                values[i] = bfs;
                valuesToDelete[i] = bfs;
            }
        }
        if (!found)
        {
            rowsIdxs.push_back(row);
            colsIdxs.push_back(col);
            values.push_back(bfs);
            valuesToDelete.push_back(bfs);
        }
        if (n_rows <= row)
            n_rows = row + 1;
        if (n_cols <= col)
            n_cols = col + 1;

    }

    /*!
     * Adds a set of basis function as diagonal features
     * \param basis the vector of basis functions
     */
    void setDiagonal(BasisFunctions_<InputC>& basis)
    {
        rowsIdxs.clear();
        colsIdxs.clear();
        clearBasis();

        int dim = basis.size();
        n_rows = n_cols = dim;

        valuesToDelete = basis;
        values = basis;

        for (int i = 0; i < dim; ++i)
        {
            rowsIdxs.push_back(i);
            colsIdxs.push_back(i);
        }
    }

    /*!
     * Destructor.
     * Destroys also all the given basis.
     */
    virtual ~SparseFeatures_()
    {
        clearBasis();
    }

private:
    void clearBasis()
    {
        for(auto basis : valuesToDelete)
        {
            delete basis;
        }

        values.clear();
        valuesToDelete.clear();
    }

private:
    std::vector<unsigned int> rowsIdxs, colsIdxs;
    BasisFunctions_<InputC> values;
    BasisFunctions_<InputC> valuesToDelete;
    unsigned int n_rows, n_cols;
};

//! Template alias.
typedef SparseFeatures_<arma::vec> SparseFeatures;

}


#endif /* INCLUDE_RELE_APPROXIMATORS_FEATURES_SPARSEFEATURES_H_ */
