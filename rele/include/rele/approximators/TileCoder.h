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
#ifndef TILECODER_H
#define TILECODER_H

#include "BasisFunctions.h"
#include "basis/Tiles.h"
#include <armadillo>

namespace ReLe
{

/**
 * @use
 * http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tiles.html
 */

class TileCoder: public AbstractBasisVector
{
protected:
    bool includeActiveFeature;
    bool useLastForHashing;
    sp_mat vector;
    int nbTilings;

public:
    TileCoder(const int& memorySize, const int& nbTilings, bool
              includeActiveFeature = true, bool useLastForHashing = false
             ) :
        includeActiveFeature(includeActiveFeature), useLastForHashing(useLastForHashing),
        vector((includeActiveFeature ? memorySize + 1 : memorySize), 1),
        nbTilings(nbTilings)
    {
    }

    virtual ~TileCoder()
    { }

    virtual void coder(const arma::vec& x, const bool& useLastAsInt) = 0;

    arma::vec operator()(const arma::vec& input)
    {
        if (includeActiveFeature)
        {
            coder(input, useLastForHashing);
            vector[vector->n_elem - 1] =  1.0;
        }
        else
            coder(input, useLastForHashing);
        return vector;
    }


    virtual double dot(const arma::vec& input, const arma::vec &otherVector)
    {
        if (includeActiveFeature)
        {
            coder(input, useLastForHashing);
            vector[vector->n_elem - 1] =  1.0;
        }
        else
            coder(input, useLastForHashing);
        double tot = vector.t() * otherVector;
        return tot;
    }

    size_t size() const
    {
        return vector.size();
    }
};


class TileCoderHashing: public TileCoder
{
private:
    typedef TileCoder Base;
    arma::vec gridResolutions;
    arma::vec inputs;
    Tiles* tiles;

public:
    TileCoderHashing(Hashing* hashing, const int& nbInputs, const T& gridResolution,
                     const int& nbTilings, const bool& includeActiveFeature = true, bool useLastForHashing = false) :
        TileCoder<T>(hashing->getMemorySize(), nbTilings, includeActiveFeature, useLastForHashing),
        gridResolutions(nbInputs), inputs(nbInputs),
        tiles(new Tiles(hashing))
    {
        gridResolutions.fill(gridResolution);
    }

    TileCoderHashing(Hashing* hashing, const int& nbInputs, arma::vec& gridResolutions,
                     const int& nbTilings, const bool& includeActiveFeature = true) :
        TileCoder(hashing->getMemorySize(), nbTilings, includeActiveFeature, useLastForHashing),
        gridResolutions(gridResolutions), inputs(nbInputs),
        tiles(new Tiles(hashing))
    { }

    virtual ~TileCoderHashing()
    {
        delete tiles;
    }

    void coder(const arma::vec& x, const bool& useLastAsInt)
    {
        Base::vector.zeros();
        inputs = x % gridResolutions;
        int dim = inputs.n_elem;
        if (useLastAsInt == true)
            tiles->tiles(&(Base::vector), Base::nbTilings, &(inputs), dim-1, inputs[dim-1]);
        else
            tiles->tiles(&(Base::vector), Base::nbTilings, &(inputs));
    }
};

} //end namespace

#endif // TILECODER