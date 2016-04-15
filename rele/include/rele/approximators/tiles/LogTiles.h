/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#ifndef INCLUDE_RELE_APPROXIMATORS_TILES_LOGTILES_H_
#define INCLUDE_RELE_APPROXIMATORS_TILES_LOGTILES_H_

#include "BasicTiles.h"

namespace ReLe
{

/*!
 * This class implements the logarithmic spaced tiling.
 * This type of tiling is a uniform tiling when considering the following
 * transformed input space:
 *
 * \f[
 * \hat{input}=\log(input - min +1)
 * \f]
 *
 * As only a finite number of tiles is supported, the state space must be range limited.
 */
class LogTiles : public BasicTiles
{
public:
    /*!
     * Constructor.
     * \param range the range of the first state variable
     * \param tilesN the number of tiles to use for the first state variable
     */
    LogTiles(const Range& range, unsigned int tilesN);

    /*!
     * Constructor.
     * \param ranges the range of the (first n) state variables
     * \param tilesN the number of tiles to use for each (of the first n) state variable
     */
    LogTiles(const std::vector<Range>& ranges,
    	     const std::vector<unsigned int>& tilesN);

    virtual unsigned int operator()(const arma::vec& input) override;
    virtual void writeOnStream(std::ostream& out) override;
    virtual void readFromStream(std::istream& in) override;

private:
    Range computeLogRange(const Range& range);
    std::vector<Range> computeLogRange(const std::vector<Range>& range);

private:
    arma::vec minComponents;
};

/*!
 * This class implements the centered logarithmic spaced tiling.
 * This type of tiling is a uniform tiling when considering the following
 * transformed input space:
 *
 * \f[
 * \hat{input}=sign(input - center)\log(|input - center| +1)
 * \f]
 *
 * As only a finite number of tiles is supported, the state space must be range limited.
 */
class CenteredLogTiles : public BasicTiles
{
public:
    /*!
     * Constructor.
     * \param range the range of the first state variable
     * \param tilesN the number of tiles to use for the first state variable
     */
    CenteredLogTiles(const Range& range, unsigned int tilesN);

    /*!
     * Constructor.
     * \param ranges the range of the (first n) state variables
     * \param tilesN the number of tiles to use for each (of the first n) state variable
     */
    CenteredLogTiles(const std::vector<Range>& ranges,
    			     const std::vector<unsigned int>& tilesN);

    virtual unsigned int operator()(const arma::vec& input) override;
    virtual void writeOnStream(std::ostream& out) override;
    virtual void readFromStream(std::istream& in) override;

private:
    Range computeLogRange(const Range& range);
    std::vector<Range> computeLogRange(const std::vector<Range>& range);

private:
    arma::vec centers;
};



}


#endif /* INCLUDE_RELE_APPROXIMATORS_TILES_LOGTILES_H_ */
