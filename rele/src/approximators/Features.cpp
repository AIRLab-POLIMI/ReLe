/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta & Marcello Restelli
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

#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"
#include "basis/GaussianRBF.h"
#include <cassert>
using namespace arma;


namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// DENSE BASIS MATRIX
///////////////////////////////////////////////////////////////////////////////////////

DenseBasisMatrix::DenseBasisMatrix(BasisFunctions& basisVector) : basis(basisVector.size())
{

    for(unsigned int i = 0; i < basisVector.size(); i++)
    {
        basis[i] = basisVector[i];
    }
}

DenseBasisMatrix::DenseBasisMatrix(BasisFunctions& basisVector, unsigned int rows, unsigned int cols)
    : basis(rows, cols)
{
    assert(rows*cols == basisVector.size());

    unsigned int k = 0;
    for(unsigned int i = 0; i < rows; i++)
    {
        for(unsigned int j = 0; j < cols; j++)
        {
            basis(i, j) = basisVector[k++];
        }
    }
}

DenseBasisMatrix::~DenseBasisMatrix()
{
    for(auto bf : basis)
    {
        delete bf;
    }
}

mat DenseBasisMatrix::operator()(const vec& input)
{
    mat output(basis.n_rows, basis.n_cols);

    //TODO use only one index???
    for(unsigned int i = 0; i < basis.n_rows; i++)
    {
        for(unsigned int j = 0; j < basis.n_cols; j++)
        {
        	BasisFunction& bf = *basis(i, j);
            output(i, j) = bf(input);
        }
    }

    return output;
}

///////////////////////////////////////////////////////////////////////////////////////
/// SPARSE BASIS MATRIX
///////////////////////////////////////////////////////////////////////////////////////

SparseBasisMatrix::SparseBasisMatrix()
    : n_rows(0), n_cols(0)
{
}

SparseBasisMatrix::SparseBasisMatrix(BasisFunctions& basis,
                                     unsigned int nbReplication, bool indipendent)
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

arma::mat SparseBasisMatrix::operator ()(const arma::vec& input)
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

void SparseBasisMatrix::addBasis(unsigned int row, unsigned int col, BasisFunction *bfs)
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



}//end namespace
