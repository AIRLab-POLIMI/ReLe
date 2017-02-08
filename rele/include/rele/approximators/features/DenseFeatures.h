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
 * This class implements a dense features vector.
 * A dense features vector is a feature vector where all the elements
 * are specified as a set of basis functions.
 * The evaluation of this features class returns a dense vector.
 */
template<class InputC>
class DenseFeatures_: public Features_<InputC>
{

public:
    /*!
     * Constructor.
     * Construct a single feature vector (a scalar).
     * \param basisFunction the basis function rappresenting this feature.
     */
    DenseFeatures_(BasisFunction_<InputC>* basisFunction, bool destroy = true)
		: basis(1), destroy(destroy)
    {
        basis[0] = basisFunction;
    }

    /*!
     * Constructor.
     * Construct a feature vector.
     * \param basisVector the set of basis functions to use
     */
    DenseFeatures_(BasisFunctions_<InputC>& basisVector, bool destroy = true)
    	:  basis(basisVector), destroy(destroy)
    {

    }

    /*!
     * Destructor.
     * Destroys also all the given basis.
     */
    virtual ~DenseFeatures_()
    {
    	if(destroy)
    	{
			for(auto bf : basis)
			{
				delete bf;
			}
    	}
    }

    virtual arma::vec operator()(const InputC& input) const override
    {
        arma::vec output(basis.size());

        for(unsigned int i = 0; i < basis.size(); i++)
        {
            BasisFunction_<InputC>& bf = *basis[i];
            output[i] = bf(input);
        }

        return output;
    }

    inline virtual size_t size() const override
    {
        return basis.size();
    }

private:
    std::vector<BasisFunction_<InputC>*> basis;
    bool destroy;

};

//! Template alias.
typedef DenseFeatures_<arma::vec> DenseFeatures;


}

#endif /* INCLUDE_RELE_APPROXIMATORS_FEATURES_DENSEFEATURES_H_ */
