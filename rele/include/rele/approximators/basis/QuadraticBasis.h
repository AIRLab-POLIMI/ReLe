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

#ifndef INCLUDE_RELE_APPROXIMATORS_BASIS_QUADRATICBASIS_H_
#define INCLUDE_RELE_APPROXIMATORS_BASIS_QUADRATICBASIS_H_

#include "BasisFunctions.h"

namespace ReLe
{

class QuadraticBasis : public BasisFunction
{
public:
    QuadraticBasis(arma::mat& Q, arma::span = arma::span::all);
    QuadraticBasis(std::vector<arma::mat>& Q);
    QuadraticBasis(std::vector<arma::mat>& Q, std::vector<arma::span> span);
    virtual double operator()(const arma::vec& input) override;
    virtual void writeOnStream(std::ostream& out) override;
    virtual void readFromStream(std::istream& in) override;

private:
    std::vector<arma::mat> Qv;
    std::vector<arma::span> spanV;
};

}

#endif /* INCLUDE_RELE_APPROXIMATORS_BASIS_QUADRATICBASIS_H_ */
