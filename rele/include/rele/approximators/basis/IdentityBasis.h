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

#ifndef INCLUDE_RELE_APPROXIMATORS_BASIS_IDENTITYBASIS_H_
#define INCLUDE_RELE_APPROXIMATORS_BASIS_IDENTITYBASIS_H_

#include "BasisFunctions.h"

namespace ReLe
{

class IdentityBasis : public BasisFunction
{
public:
    IdentityBasis(unsigned int index);
    virtual ~IdentityBasis();
    double operator() (const arma::vec& input);


    virtual void writeOnStream (std::ostream& out);
    virtual void readFromStream(std::istream& in);

    static BasisFunctions generate(unsigned int input_size);

private:
    unsigned int index;
};

}


#endif /* INCLUDE_RELE_APPROXIMATORS_BASIS_IDENTITYBASIS_H_ */
