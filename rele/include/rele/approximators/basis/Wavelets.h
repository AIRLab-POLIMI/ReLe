/*
 * rele,
 *
 *
 * Copyright (C) 2017 Davide Tateo
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

#ifndef INCLUDE_RELE_APPROXIMATORS_BASIS_WAVELETS_H_
#define INCLUDE_RELE_APPROXIMATORS_BASIS_WAVELETS_H_

/*
 * rele,
 *
 *
 * Copyright (C) 2017 Davide Tateo
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

#include "rele/approximators/BasisFunctions.h"
#include <armadillo>

namespace ReLe
{

class WaveletType
{
public:
    virtual double scaling(double value) = 0;
    virtual double mother(double value) = 0;

    virtual ~WaveletType()
    {

    }

};

/*!
 * This class implements a generic wavelets basis function with given scale and translation.
 */
class Wavelets : public BasisFunction
{

public:
    /*!
     * Constructor.
     * Construct a scaling function
     */
    Wavelets(WaveletType& wavelet, unsigned int k, unsigned int index);

    /*!
     * Constructor.
     * Construct the jk-th wavelet
     * \param j the order
     * \param phi the number
     * \param index the input component to be processed
     */
    Wavelets(WaveletType& wavelet, unsigned int j, unsigned int k, unsigned int index);

    /*!
     * Destructor.
     */
    virtual ~Wavelets();

    double operator() (const arma::vec& input) override;

    /*!
     * Return the set of wavelets and scaling functions
     * \param jMax the maximum scaling factor
     * \param kMax the maximum translation length
     * \return the generated basis functions
     */
    static BasisFunctions generate(WaveletType& wavelet, unsigned int index, unsigned int jMax, int maxT);

    virtual void writeOnStream (std::ostream& out) override;
    virtual void readFromStream(std::istream& in) override;

private:
    bool scale;
    unsigned int j;
    unsigned int k;
    unsigned int index;
    WaveletType& wavelet;

};

}//end namespace


#endif /* INCLUDE_RELE_APPROXIMATORS_BASIS_WAVELETS_H_ */
