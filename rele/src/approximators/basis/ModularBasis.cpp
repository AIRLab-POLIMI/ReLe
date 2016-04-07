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

#include "rele/approximators/basis/ModularBasis.h"

namespace ReLe
{

ModularBasis::ModularBasis(unsigned int index1, unsigned int index2, const ModularRange& range)
    : index1(index1), index2(index2), range(range)
{

}


ModularSum::ModularSum(unsigned int index1, unsigned int index2, const ModularRange& range)
    : ModularBasis(index1, index2, range)
{

}

double ModularSum::operator()(const arma::vec& input)
{
    return range.bound(input(index1) + input(index2));
}

void ModularSum::writeOnStream(std::ostream& out)
{
    out << "Modular Sum Basis {" << index1 << "," <<index2 << "} on" << range << std::endl;
}

void ModularSum::readFromStream(std::istream& in)
{
    //TODO [SERIALIZATION] implement
}


ModularDifference::ModularDifference(unsigned int index1, unsigned int index2, const ModularRange& range)
    : ModularBasis(index1, index2, range)
{

}

double ModularDifference::operator()(const arma::vec& input)
{
    return range.bound(input(index1) - input(index2));
}

void ModularDifference::writeOnStream(std::ostream& out)
{
    out << "Modular Difference Basis {" << index1 << "," <<index2 << "} on" << range << std::endl;
}

void ModularDifference::readFromStream(std::istream& in)
{
    //TODO [SERIALIZATION] implement
}


ModularProduct::ModularProduct(unsigned int index1, unsigned int index2, const ModularRange& range)
    : ModularBasis(index1, index2, range)
{

}

double ModularProduct::operator()(const arma::vec& input)
{
    return range.bound(input(index1) * input(index2));
}

void ModularProduct::writeOnStream(std::ostream& out)
{
    out << "Modular Product Basis {" << index1 << "," <<index2 << "} on" << range << std::endl;
}

void ModularProduct::readFromStream(std::istream& in)
{
    //TODO [SERIALIZATION] implement
}


ModularDivision::ModularDivision(unsigned int index1, unsigned int index2, const ModularRange& range)
    : ModularBasis(index1, index2, range)
{

}

double ModularDivision::operator()(const arma::vec& input)
{
    return range.bound(input(index1) / input(index2));
}

void ModularDivision::writeOnStream(std::ostream& out)
{
    out << "Modular Division Basis {" << index1 << "," <<index2 << "} on" << range << std::endl;
}

void ModularDivision::readFromStream(std::istream& in)
{
    //TODO [SERIALIZATION] implement
}

}

