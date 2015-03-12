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

DenseBasisVector::DenseBasisVector()
{
}

DenseBasisVector::~DenseBasisVector()
{
    DenseBasisVector::iterator it;
    for (it = this->begin(); it != this->end(); ++it)
    {
        delete *it;
    }
    this->clear();
}

mat DenseBasisVector::operator()(const vec& input)
{
    vec output(this->size());

    unsigned int i = 0;
    std::vector<BasisFunction*>::iterator it;
    for (it = this->begin(); it != this->end(); ++it)
    {
        BasisFunction& bf = *(*it);
        output[i++] = bf(input);
    }

    return output;
}

double DenseBasisVector::dot(const vec& input, const vec& otherVector)
{
    assert(this->size() == otherVector.size());

    double dotprod = 0.0;
    unsigned int i = 0;

    std::vector<BasisFunction*>::iterator it;
    for (it = this->begin(); it != this->end(); ++it)
    {
        BasisFunction& bf = *(*it);
        dotprod += otherVector[i++] * bf(input);
    }

    return dotprod;
}

void
DenseBasisVector::generatePolynomialBasisFunctions(unsigned int degree, unsigned int input_size)
{
    //  AddBasisFunction(new PolynomialFunction(0,0));
    std::vector<unsigned int> dim;
    for (unsigned int i = 0; i < input_size; i++)
    {
        dim.push_back(i);
    }
    for (unsigned int d = 0; d <= degree; d++)
    {
        std::vector<unsigned int> deg(input_size);
        deg[0] = d;
        generatePolynomialsPermutations(deg, dim);
        generatePolynomials(deg, dim, 1);
    }
    std::cout << size() << " polynomial basis functions added!" << std::endl;
}


void DenseBasisVector::display(std::vector<unsigned int> v)
{
    for (vector<unsigned int >::iterator it = v.begin(); it != v.end(); ++it)
    {
        std::cout << *it;
    }
    std::cout << std::endl;
}

void DenseBasisVector::generatePolynomialsPermutations(vector<unsigned int> deg,
        vector<unsigned int>& dim)
{
    std::sort(deg.begin(), deg.end());
    do
    {
        BasisFunction* pBF = new PolynomialFunction(dim, deg);
        this->push_back(pBF);
        //    display(deg);
    }
    while (next_permutation(deg.begin(), deg.end()));
}

void DenseBasisVector::generatePolynomials(vector<unsigned int> deg,
        vector<unsigned int>& dim,
        unsigned int place)
{
    if (deg.size() > 1)
    {
        if (deg[0] > deg[1] && deg[place] < deg[place - 1] && deg[0] - deg[place] > 1)
        {
            std::vector<unsigned int> degree = deg;
            degree[0]--;
            degree[place]++;
            generatePolynomialsPermutations(degree, dim);
            generatePolynomials(degree, dim, place);
            if (place < deg.size() - 1)
            {
                generatePolynomials(degree, dim, place + 1);
            }
        }
    }
}

std::ostream& operator<< (std::ostream& out, DenseBasisVector& bf)
{
    out << bf.size() << std::endl;
    for (unsigned int i = 0; i < bf.size(); i++)
    {
        bf[i]->writeOnStream(out);
        out << std::endl;
    }
    return out;
}

std::istream& operator>> (std::istream& in, DenseBasisVector& bf)
{
    unsigned int num_basis_functions;
    in >> num_basis_functions;
    for (unsigned int i = 0; i < num_basis_functions; i++)
    {
        std::string type;
        in >> type;
        BasisFunction* function = 0;
        if (type == "Polynomial")
        {
            function = new PolynomialFunction();
        }
        else if (type == "GaussianRbf")
        {
            function = new GaussianRbf();
        }
        else
        {
            std::cerr << "ERROR: Unrecognized basis-function type" << std::endl;
            exit(1);
        }
        //        in >> *function;
        function->readFromStream(in);
        bf.push_back(function);
    }
    return in;
}

///////////////////////////////////////////////////////////////////////////////////////
/// IDENTITY BASIS
///////////////////////////////////////////////////////////////////////////////////////

IdentityBasis::~IdentityBasis()
{
}

arma::mat IdentityBasis::operator()(const arma::vec& input)
{
    return input;
}

double IdentityBasis::dot(const arma::vec& input, const arma::vec& otherVector)
{
    arma::vec res = input.t() * otherVector;
    return res(0);
}

size_t IdentityBasis::size() const
{
    return stateSize;
}

arma::vec IdentityBasis::operator()(size_t input)
{
    arma::vec phi(stateSize, arma::fill::zeros);
    phi(input) = 1;
    return phi;
}

double IdentityBasis::dot(size_t input, const arma::vec& otherVector)
{
    return otherVector[input];
}

///////////////////////////////////////////////////////////////////////////////////////
/// SPARSE BASIS MATRIX
///////////////////////////////////////////////////////////////////////////////////////

SparseBasisMatrix::SparseBasisMatrix()
    : n_rows(0), n_cols(0)
{
}

SparseBasisMatrix::SparseBasisMatrix(DenseBasisVector& basis,
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
