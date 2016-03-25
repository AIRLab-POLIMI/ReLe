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

#include "rele/approximators/BasisFunctions.h"
#include "rele/core/BasicFunctions.h"

namespace ReLe
{

/*!
 * Trait defined to return sparse or dense features matrices
 */
template<bool denseOutput>
struct feature_traits
{

};

template<>
struct feature_traits<true>
{
    typedef arma::mat type;
    typedef arma::vec column_type;
};

template<>
struct feature_traits<false>
{
    typedef arma::sp_mat type;
    typedef arma::sp_vec column_type;
};

/*!
 * This interface represent a generic set of features.
 * Features are a mapping \f$\phi(i)\rightarrow\mathbb{R}^n\times\mathbb{R}^m\f$ with \f$i\in\mathcal{D}\f$.
 * The template definition allows for generic domains \f$\mathcal{D}\f$.
 * Optionally, the features matrix returned can be a sparse matrix, this is done by setting to false the optional
 * template parameter denseOutput.
 */
template<class InputC, bool denseOutput = true>
class Features_
{
    using return_type = typename feature_traits<denseOutput>::type;

public:

    /*!
     * Destructor.
     */
    virtual ~Features_()
    {
    }

    /*!
     * Evaluate the features in input
     * \param input the input data
     * \return the evaluated features
     */
    virtual return_type operator()(const InputC& input) = 0;

    /*!
     * Overloading of the evaluation method, simply vectorizes the inputs and then evaluates
     * the feature with the default 1 input method
     * \param input1 the first input data
     * \param input2 the second input data
     * \return the evaluated features
     */
    template<class Input1, class Input2>
    return_type operator()(const Input1& input1, const Input2& input2)
    {
        auto& self = *this;
        return self(vectorize(input1, input2));
    }

    /*!
     * Overloading of the evaluation method, simply vectorizes the inputs and then evaluates
     * the feature with the default 1 input method
     * \param input1 the first input data
     * \param input2 the second input data
     * \param input3 the third input data
     * \return the evaluated features
     */
    template<class Input1, class Input2, class Input3>
    return_type operator()(const Input1& input1, const Input2& input2, const Input3& input3)
    {
        auto& self = *this;
        return self(vectorize(input1, input2, input3));
    }

    /*!
     * Getter.
     * \return the number of rows of the output feature matrix
     */
    virtual size_t rows() const = 0;

    /*!
     * Getter.
     * \return the number of columns of the output feature matrix
     */
    virtual size_t cols() const = 0;

};

//! Template alias.
typedef Features_<arma::vec> Features;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_FEATURES_H_ */
