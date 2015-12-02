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

#ifndef INCLUDE_RELE_APPROXIMATORS_BATCHREGRESSORTRAITS_H_
#define INCLUDE_RELE_APPROXIMATORS_BATCHREGRESSORTRAITS_H_

#include "Basics.h"

namespace ReLe
{
template<class InputC>
struct input_collection
{
    typedef arma::field<InputC> type;
    typedef typename std::add_lvalue_reference<arma::field<InputC>>::type ref_type;
    typedef typename std::add_lvalue_reference<const arma::field<InputC>>::type const_ref_type;
};

template<>
struct input_collection<size_t>
{
    typedef arma::uvec type;
    typedef typename std::add_lvalue_reference<arma::uvec>::type ref_type;
    typedef typename std::add_lvalue_reference<const arma::uvec>::type const_ref_type;
};

template<>
struct input_collection<arma::vec>
{
    typedef arma::mat type;
    typedef typename std::add_lvalue_reference<arma::mat>::type ref_type;
    typedef typename std::add_lvalue_reference<const arma::mat>::type const_ref_type;
};

template<class OutputC>
struct output_collection
{
    typedef arma::field<OutputC> type;
    typedef typename std::add_lvalue_reference<arma::field<OutputC>>::type ref_type;
    typedef typename std::add_lvalue_reference<const arma::field<OutputC>>::type const_ref_type;
};

template<>
struct output_collection<unsigned int>
{
    typedef arma::uvec type;
    typedef typename std::add_lvalue_reference<arma::uvec>::type ref_type;
    typedef typename std::add_lvalue_reference<const arma::uvec>::type const_ref_type;
};

template<>
struct output_collection<arma::vec>
{
    typedef arma::mat type;
    typedef typename std::add_lvalue_reference<arma::mat>::type ref_type;
    typedef typename std::add_lvalue_reference<const arma::mat>::type const_ref_type;
};


}

#endif /* INCLUDE_RELE_APPROXIMATORS_BATCHREGRESSORTRAITS_H_ */
