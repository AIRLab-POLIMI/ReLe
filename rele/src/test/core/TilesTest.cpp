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

#include "features/TilesCoder.h"
#include "tiles/BasicTiles.h"

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    cout << "## Tiles Test ##" << endl;

    //Ranges
    Range range(0.0, 2.0);

    std::vector<Range> ranges;
    ranges.push_back(range);
    ranges.push_back(range);
    std::vector<unsigned int> tilesN1;
    tilesN1.resize(2, 2);
    std::vector<unsigned int> tilesN2;
    tilesN2.resize(2, 3);


    //Tiles
    Tiles* tiles[6];
    tiles[0] = new BasicTiles(range, 10);
    tiles[1] = new BasicTiles(range, 10);
    tiles[2] = new BasicTiles(ranges, tilesN1);
    tiles[3] = new BasicTiles(ranges, tilesN2);
    tiles[4] = new BasicTiles(ranges, tilesN1);
    tiles[5] = new BasicTiles(ranges, tilesN2);

    for(unsigned int i = 0; i < 6; i++)
    {
        cout << *tiles[i] << endl;
    }

    //Inputs
    arma::vec input1(1);
    input1(0) = 0.55;

    arma::vec input2(2);
    input2(0) = 1.55;
    input2(1) = 1.67;

    arma::vec input3(1);
    input3(0) = 1;

    arma::vec input4(1);
    input4(0) = -100;

    cout << "## Single tiling Test ##" << endl;
    TilesCoder phi1(tiles[2]);

    cout << "F(" << input2[0] << "," << input2[1] << ") = " << endl;
    cout << arma::mat(phi1(input2)) << endl;

}
