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

#ifndef INCLUDE_RELE_APPROXIMATORS_BASIS_IDENTITYBASIS_H_
#define INCLUDE_RELE_APPROXIMATORS_BASIS_IDENTITYBASIS_H_

#include "rele/approximators/BasisFunctions.h"

namespace ReLe
{

/*!
 * This template class implements functions to build basis functions that replicates the input.
 */
template<class InputC>
class IdentityBasis_ : public BasisFunction_<InputC>
{
public:
    /*!
     * Constructor.
     * \param index the index of the element in the input
     */
    IdentityBasis_(unsigned int index) : index(index)
    {

    }

protected:
    unsigned int index;
};

/*!
 * This class implements functions to build basis functions that replicates the input vector.
 */
class IdentityBasis : public IdentityBasis_<arma::vec>
{
public:
    /*!
     * Constructor.
     * \param index the index of the element in the input
     */
    IdentityBasis(unsigned int index);

    /*!
     * Destructor.
     */
    virtual ~IdentityBasis();

    double operator() (const arma::vec& input) override;
    virtual void writeOnStream (std::ostream& out) override;
    virtual void readFromStream(std::istream& in) override;

    /*!
     * Return the input in the form of a Basis Functions.
     * \param input_size the size of the input vector
     * \return the generated basis functions
     */
    static BasisFunctions generate(unsigned int input_size);
};

/*!
 * This class implements functions to build basis functions that replicates the input vector with
 * finite values.
 */
class FiniteIdentityBasis : public IdentityBasis_<size_t>
{
public:
    /*!
     * Constructor.
     * \param index the index of the element in the input
     */
    FiniteIdentityBasis(unsigned int index);

    /*!
     * Destructor.
     */
    virtual ~FiniteIdentityBasis();

    double operator() (const size_t& input) override;
    virtual void writeOnStream (std::ostream& out) override;
    virtual void readFromStream(std::istream& in) override;

    /*!
     * Return the value associated to the given input.
     * \param stateN the input value
     * \return the generated basis functions
     */
    static BasisFunctions_<size_t> generate(unsigned int stateN);
};

/*!
 * This class implements functions to build basis functions that associate a value to each finite
 * value in the input vector.
 */
class VectorFiniteIdentityBasis : public IdentityBasis_<arma::vec>
{
public:
    /*!
     * Constructor.
     * \param index the index of the element in the input
     * \param value the value to associate at the input element in the given index
     */
    VectorFiniteIdentityBasis(unsigned int index, double value);

    /*!
     * Destructor.
     */
    virtual ~VectorFiniteIdentityBasis();

    double operator() (const arma::vec& input) override;
    virtual void writeOnStream (std::ostream& out) override;
    virtual void readFromStream(std::istream& in) override;

    /*!
     * Return the values associated to each input.
     * \param values the vector of values
     * \return the generated basis functions
     */
    static BasisFunctions generate(std::vector<unsigned int> values);

    /*!
     * Return the value associated to the given input.
     * \param stateN the input value
     * \param values the vector of values
     * \return the generated basis functions
     */
    static BasisFunctions generate(unsigned int stateN, unsigned int values);

private:
    double value;
};

/*!
 * This class implements functions to build basis functions that, given an input value, return
 * its inverse value.
 */
template<class InputC>
class InverseBasis_ : public BasisFunction_<InputC>
{
public:
    /*!
     * Constructor.
     * \param basis basis function
     */
    InverseBasis_(BasisFunction_<InputC>* basis) : basis(basis)
    {

    }

    /*!
     * Destructor.
     */
    ~InverseBasis_()
    {
        delete basis;
    }

    double operator() (const InputC& input) override
    {
        BasisFunction_<InputC>& bf = *basis;
        return -bf(input);
    }

    void writeOnStream (std::ostream& out) override
    {
        out << "Inverse ";
        basis->writeOnStream(out);
    }

    void readFromStream(std::istream& in) override
    {
        //TODO [SERIALIZATION] implement
    }

    /*!
     * Return the inverse value associated to the given input.
     * \param basis the basis function whose output value is inverted
     * \return the generated basis functions
     */
    static BasisFunctions_<InputC> generate(BasisFunctions_<InputC> basis)
    {
        BasisFunctions_<InputC> newBasis;

        for(auto bf : basis)
        {
            newBasis.push_back(new InverseBasis_<InputC>(bf));
        }

        return newBasis;
    }

private:
    BasisFunction_<InputC>* basis;
};

typedef InverseBasis_<arma::vec> InverseBasis;

}


#endif /* INCLUDE_RELE_APPROXIMATORS_BASIS_IDENTITYBASIS_H_ */
