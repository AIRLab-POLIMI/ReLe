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

#ifndef INCLUDE_RELE_APPROXIMATORS_BASIS_FREQUENCYBASIS_H_
#define INCLUDE_RELE_APPROXIMATORS_BASIS_FREQUENCYBASIS_H_

#include "rele/approximators/BasisFunctions.h"
#include <armadillo>

namespace ReLe
{

/*!
 * This class implements a sinusoid basis function with given frequency and phase.
 */
class FrequencyBasis : public BasisFunction
{

public:
    /*!
     * Constructor.
     * \param f the frequency of the sinusoid
     * \param phi the phase of the sinusoid
     * \param index the input component to be processed
     */
	FrequencyBasis(double f, double phi, unsigned int index);

    /*!
     * Destructor.
     */
    virtual ~FrequencyBasis();

    double operator() (const arma::vec& input) override;

    /*!
     * Getter.
     * \return frequency of the sinusoid
     */
    inline double getFrequency()
    {
        return omega/(2.0*M_PI);
    }

    /*!
     * Getter.
     * \return phase of the sinusoid
     */
    inline double getPhase()
    {
        return phi;
    }

    /*!
     * Return the set of sinusoids basis functions with phase phi, starting from the frequency
     * fS to the frequency fE, with freqeuncy step of df.
     * \param index the component to be considered from this basis
     * \param fS starting sinusoid frequency
     * \param fE final sinusoid frequency
     * \param df step to take to the next frequency
     * \param phi sinusoid phase
     * \return the generated basis functions
     */
    static BasisFunctions generate(unsigned int index, double fS, double fE, double df, double phi);

    /*!
     * Return the set of sinusoids basis functions (sine or cosines), starting from the frequency
     * fS to the frequency fE, with freqeuncy step of df.
     * \param index the component to be considered from this basis
     * \param fS starting sinusoid frequency
     * \param fE final sinusoid frequency
     * \param df step to take to the next frequency
     * \param sine if using a sine or cosine sinusoid
     * \return the generated basis functions
     */
    static BasisFunctions generate(unsigned int index, double fS, double fE, double df, bool sine = true);

    virtual void writeOnStream (std::ostream& out) override;
    virtual void readFromStream(std::istream& in) override;

private:
    double omega;
    double phi;
    unsigned int index;

};

}//end namespace


#endif /* INCLUDE_RELE_APPROXIMATORS_BASIS_FREQUENCYBASIS_H_ */
