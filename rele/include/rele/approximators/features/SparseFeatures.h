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

#ifndef INCLUDE_RELE_APPROXIMATORS_FEATURES_SPARSEFEATURES_H_
#define INCLUDE_RELE_APPROXIMATORS_FEATURES_SPARSEFEATURES_H_

#include "Features.h"

namespace ReLe
{

template<class InputC>
class SparseFeatures_: public Features_<InputC>
{

public:
    SparseFeatures_(): n_rows(0), n_cols(0)
    {
    }

    SparseFeatures_(BasisFunctions_<InputC>& basis,
                    unsigned int nbReplication = 1,
                    bool indipendent = true)
        : n_rows(basis.size()*(indipendent?nbReplication:1)), n_cols(nbReplication)
    {
        unsigned int i, k;
        unsigned int offset = 0;
        unsigned int nBasis = basis.size();
        for (k = 0; k < nbReplication; ++k)
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

    arma::mat operator ()(const InputC& input)
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

    inline size_t rows() const
    {
        return n_rows;
    }

    size_t cols() const
    {
        return n_cols;
    }

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
            }
        }
        if (!found)
        {
            rowsIdxs.push_back(row);
            colsIdxs.push_back(col);
            values.push_back(bfs);
        }
        if (n_rows <= row)
            n_rows = row + 1;
        if (n_cols <= col)
            n_cols = col + 1;

    }

private:
    //an element of the matrix is given by (rowsIdxs[i], colsIdxs[i], values[i])
    std::vector<unsigned int> rowsIdxs, colsIdxs;
    BasisFunctions_<InputC> values;
    unsigned int n_rows, n_cols;
};


typedef SparseFeatures_<arma::vec> SparseFeatures;

}


#endif /* INCLUDE_RELE_APPROXIMATORS_FEATURES_SPARSEFEATURES_H_ */
