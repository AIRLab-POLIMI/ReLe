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

#ifndef INCLUDE_RELE_APPROXIMATORS_FEATURES_H_
#define INCLUDE_RELE_APPROXIMATORS_FEATURES_H_

#include "BasisFunctions.h"

namespace ReLe
{

template<bool denseOutput>
struct feature_traits
{

};

template<>
struct feature_traits<true>
{
    typedef arma::mat type;
};

template<>
struct feature_traits<false>
{
    typedef arma::sp_mat type;
};

template<class InputC, bool denseOutput = true>
class Features_
{
    using return_type = typename feature_traits<denseOutput>::type;

public:

    virtual ~Features_()
    {
    }

    virtual return_type operator()(const InputC& input) = 0;
    virtual size_t rows() const = 0;
    virtual size_t cols() const = 0;

    void setDiagonal(BasisFunctions& basis);

};


typedef Features_<arma::vec> Features;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_FEATURES_H_ */
