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

#include "basis/QuadraticBasis.h"

#include "CSV.h"

#include <cassert>


namespace ReLe
{

QuadraticBasis::QuadraticBasis(arma::mat& Q, arma::span span)
{
    Qv.push_back(Q);
    spanV.push_back(span);
}

QuadraticBasis::QuadraticBasis(std::vector<arma::mat>& Q) : Qv(Q), spanV(Qv.size(), arma::span::all)
{
}

QuadraticBasis::QuadraticBasis(std::vector<arma::mat>& Q, std::vector<arma::span> span) : Qv(Q), spanV(span)
{
    assert(Qv.size() > 0);
}

double QuadraticBasis::operator()(const arma::vec& input)
{
    arma::mat J(1, 1, arma::fill::zeros);

    for(int i = 0; i < Qv.size(); i++)
    {
    	const arma::vec& x = input(spanV[i]);
    	J += x.t()*Qv[i]*x;
    }

    return J[0];
}

void QuadraticBasis::writeOnStream(std::ostream& out)
{
    out << "Quadratic Basis" << std::endl;
    for(const auto& Q : Qv)
    {
        out << "---" << std::endl;
        CSVutils::matrixToCSV(Q, out);
    }
}

void QuadraticBasis::readFromStream(std::istream& in)
{

}

}

