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

#ifndef BASICSTRAITS_H_
#define BASICSTRAITS_H_

#include <type_traits>

#include "Basics.h"


namespace ReLe
{
template<class ActionC>
struct action_type
{
    typedef typename std::add_pointer<void>::type type;
    typedef typename std::add_pointer<const void>::type const_type;
    typedef typename std::add_pointer<void>::type type_ref;
    typedef const std::add_pointer<const void>::type const_type_ref;
};

template<>
struct action_type<FiniteAction>
{
    typedef unsigned int type;
    typedef const unsigned int const_type;
    typedef typename std::add_lvalue_reference<unsigned int>::type type_ref;
    typedef typename std::add_lvalue_reference<const unsigned int>::type const_type_ref;
};

template<>
struct action_type<DenseAction>
{
    typedef arma::vec type;
    typedef const arma::vec const_type;
    typedef typename std::add_lvalue_reference<arma::vec>::type type_ref;
    typedef typename std::add_lvalue_reference<const arma::vec>::type const_type_ref;
};

template<class StateC>
struct state_type
{
    typedef typename std::add_pointer<void>::type type;
    typedef typename std::add_pointer<const void>::type const_type;
    typedef typename std::add_pointer<void>::type type_ref;
    typedef const std::add_pointer<const void>::type const_type_ref;
};

template<>
struct state_type<FiniteState>
{
    typedef size_t type;
    typedef const size_t const_type;
    typedef typename std::add_lvalue_reference<size_t>::type type_ref;
    typedef typename std::add_lvalue_reference<const size_t>::type const_type_ref;
};

template<>
struct state_type<DenseState>
{
    typedef arma::vec type;
    typedef const arma::vec const_type;
    typedef typename std::add_lvalue_reference<arma::vec>::type type_ref;
    typedef typename std::add_lvalue_reference<const arma::vec>::type const_type_ref;
};

}

#endif /* BASICSTRAITS_H_ */
