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

#ifndef INCLUDE_RELE_APPROXIMATORS_TILES_BASICTILES_H_
#define INCLUDE_RELE_APPROXIMATORS_TILES_BASICTILES_H_

#include "rele/approximators/Tiles.h"
#include "rele/utils/Range.h"

namespace ReLe
{

/*!
 * This class implements the most simple type of tiling: a uniform grid tiling
 * over the whole state space.
 * As only a finite number of tiles is supported, the state space must be range limited.
 */
class BasicTiles : public Tiles
{
public:
    /*!
     * Constructor.
     * \param range the range of the first state variable
     * \param tilesN the number of tiles to use for the first state variable
     */
    BasicTiles(Range& range, unsigned int tilesN);

    /*!
     * Constructor.
     * \param ranges the range of the (first n) state variables
     * \param tilesN the number of tiles to use for each (of the first n) state variable
     */
    BasicTiles(std::vector<Range>& ranges, std::vector<unsigned int>& tilesN);

    inline virtual unsigned int size() override
    {
        return tilesSize;
    }

    virtual unsigned int operator()(const arma::vec& input) override;
    virtual void writeOnStream(std::ostream& out) override;
    virtual void readFromStream(std::istream& in) override;

protected:
    unsigned int computeComponentIndex(unsigned int i,double value);

private:
    void computeSize();

protected:
    std::vector<Range> ranges;
    std::vector<unsigned int> tilesN;
    unsigned int tilesSize;
};

/*!
 * This class extends the BasicTiles class, allowing to choose over which state component
 * tiling should be applied.
 */
class SelectiveTiles : public BasicTiles
{
public:
    /*!
     * Constructor.
     * \param stateComponents the index of the state variables to use
     * \param ranges the range to use for each state variable
     * \param tilesN the number of tiles to use for each state variable
     */
    SelectiveTiles(std::vector<unsigned int> stateComponents,
                   std::vector<Range>& ranges, std::vector<unsigned int>& tilesN);

    virtual unsigned int operator()(const arma::vec& input) override;
    virtual void writeOnStream(std::ostream& out) override;
    virtual void readFromStream(std::istream& in) override;

private:
    std::vector<unsigned int> stateComponents;
};



}


#endif /* INCLUDE_RELE_APPROXIMATORS_TILES_BASICTILES_H_ */
