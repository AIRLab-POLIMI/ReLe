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
    Range range(0.0, 1.0);

    std::vector<Range> ranges;
    ranges.push_back(range);
    ranges.push_back(range);
    std::vector<unsigned int> tilesN1;
    tilesN1.resize(2, 2);
    std::vector<unsigned int> tilesN2;
    tilesN2.resize(2, 3);


    //Tiles
    Tiles* tiles1 = new BasicTiles(range, 10);
    Tiles* tiles2 = new BasicTiles(range, 10);
    Tiles* tiles3 = new BasicTiles(ranges, tilesN1);
    Tiles* tiles4 = new BasicTiles(ranges, tilesN2);
    Tiles* tiles5 = new BasicTiles(ranges, tilesN1);
    Tiles* tiles6 = new BasicTiles(ranges, tilesN2);


    //Inputs
    arma::vec input1(1);
    input1(0) = 0.55;

    arma::vec input2(2);
    input2(0) = 0.55;
    input2(1) = 0.67;

    cout << "## Single tiling Test ##" << endl;
    TilesCoder phi1(tiles1);



    cout << "F(" << input1[0] << ") = " << endl;
    cout << arma::mat(phi1(input1)) << endl;

    cout << "## Single tiling double output Test ##" << endl;
    TilesCoder phi2(tiles2, 2);

    cout << "F(" << input1[0] << ") = " << endl;
    cout << arma::mat(phi2(input1)) << endl;

    cout << "## Double tiling single output Test ##" << endl;
    TilesVector tVector0;
    tVector0.push_back(tiles3);
    tVector0.push_back(tiles4);
    TilesCoder phi3(tVector0, 1);

    cout << "F(" << input2[0] << ", " << input2[1] << ") = " << endl;
    cout << arma::mat(phi3(input2)) << endl;

    cout << "## Double tiling double output Test ##" << endl;
    TilesVector tVector1;
    tVector1.push_back(tiles5);
    tVector1.push_back(tiles6);
    TilesCoder phi4(tVector1, 2);

    cout << "F(" << input2[0] << ", " << input2[1] << ") = " << endl;
    cout << arma::mat(phi4(input2)) << endl;



}
