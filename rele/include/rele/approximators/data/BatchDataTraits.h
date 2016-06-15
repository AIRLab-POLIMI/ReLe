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

#ifndef INCLUDE_RELE_APPROXIMATORS_BATCHDATATRAITS_H_
#define INCLUDE_RELE_APPROXIMATORS_BATCHDATATRAITS_H_

#include "rele/core/Basics.h"

#include <type_traits>

namespace ReLe
{

/* Traits to handle input */
template<bool denseOutput>
struct input_traits
{

};

template<>
struct input_traits<true>
{
    typedef arma::mat type;
    typedef arma::mat& ref_type;
    typedef const arma::mat& const_ref_type;
    typedef arma::vec column_type;
};

template<>
struct input_traits<false>
{
    typedef arma::sp_mat type;
    typedef arma::sp_mat& ref_type;
    typedef const arma::sp_mat& const_ref_type;
    typedef arma::sp_vec column_type;
};

/* Traits to handle output */
template<class OutputC>
struct output_traits
{
    typedef arma::field<OutputC> type;
    typedef typename std::add_lvalue_reference<arma::field<OutputC>>::type ref_type;
    typedef typename std::add_lvalue_reference<const arma::field<OutputC>>::type const_ref_type;

    static arma::mat square(const OutputC& o)
    {
        return arma::mat();
    }

    static bool isAlmostEqual(const OutputC& o1, const OutputC& o2)
    {
        return o1 == o2;
    }

    static double errorSquared(const OutputC& o1, const OutputC& o2)
    {
        return 0;
    }

};

template<>
struct output_traits<arma::vec>
{
    typedef arma::mat type;
    typedef typename std::add_lvalue_reference<arma::mat>::type ref_type;
    typedef typename std::add_lvalue_reference<const arma::mat>::type const_ref_type;

    static arma::mat square(const arma::vec& o)
    {
        return o*o.t();
    }

    static bool isAlmostEqual(const arma::vec& o1, const arma::vec& o2)
    {
        return arma::norm(o1 - o2) < 1e-7;
    }

    static double errorSquared(const arma::vec& o1, const arma::vec& o2)
    {
        arma::vec e = o1 - o2;
        return arma::as_scalar(e.t()*e);
    }

};

template<>
struct output_traits<unsigned int>
{
    typedef arma::uvec type;
    typedef typename std::add_lvalue_reference<arma::uvec>::type ref_type;
    typedef typename std::add_lvalue_reference<const arma::uvec>::type const_ref_type;

    static arma::mat square(const unsigned int& o)
    {
        arma::mat m;
        m = o*o;
        return m;
    }

    static bool isAlmostEqual(const unsigned int& o1, const unsigned int& o2)
    {
        return o1 == o2;
    }

    static double errorSquared(const unsigned int& o1, const unsigned int& o2)
    {
        double e = static_cast<double>(o1) - o2;
        return e*e;
    }
};



}

#endif /* INCLUDE_RELE_APPROXIMATORS_BATCHDATATRAITS_H_ */
