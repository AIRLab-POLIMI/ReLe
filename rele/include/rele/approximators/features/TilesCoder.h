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

#ifndef INCLUDE_RELE_APPROXIMATORS_FEATURES_TILESCODER_H_
#define INCLUDE_RELE_APPROXIMATORS_FEATURES_TILESCODER_H_

#include "rele/approximators/Features.h"
#include "rele/approximators/Tiles.h"

namespace ReLe
{

template<class InputC>
class TilesCoder_: public Features_<InputC, false>
{
public:
    TilesCoder_(Tiles_<InputC>* tiles, unsigned int outputs = 1) :
        outputs(outputs)
    {
        tilesVector.push_back(tiles);
        rowsN = tiles->size() * outputs;
    }

    TilesCoder_(TilesVector_<InputC>& tilesVector, unsigned int outputs = 1) :
        tilesVector(tilesVector), outputs(outputs), rowsN(0)
    {
        computeRows();
    }

    virtual arma::sp_mat operator()(const InputC& input) override
    {
        arma::sp_mat output(rowsN, outputs);

        unsigned int offset = 0;
        for(unsigned int o = 0; o < outputs; o++)
        {
            for(auto t : tilesVector)
            {
                auto& tiles = *t;
                computeTile(input, offset, o, tiles, output);
                offset += tiles.size();
            }
        }

        return output;
    }

    virtual size_t rows() const override
    {
        return rowsN;
    }

    virtual size_t cols() const override
    {
        return outputs;
    }

    virtual ~TilesCoder_()
    {
        for(auto tile : tilesVector)
            delete tile;
    }

protected:
    void computeTile(const InputC& input, unsigned int offset,
                     unsigned int o, Tiles_<InputC>& tiles,
                     arma::sp_mat& output)
    {
        try
        {
            unsigned int index = tiles(input) + offset;
            output(index, o) = 1.0;
        }
        catch (out_of_bounds& e) {}

    }

private:
    void computeRows()
    {
        rowsN = 0;

        for (auto& tiles : tilesVector)
        {
            rowsN += tiles->size();
        }

        rowsN *= outputs;
    }

private:
    TilesVector_<InputC> tilesVector;
    unsigned int outputs;
    size_t rowsN;
};

typedef TilesCoder_<arma::vec> TilesCoder;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_FEATURES_TILESCODER_H_ */
