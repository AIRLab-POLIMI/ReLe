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

#include "rele/approximators/features/TilesCoder.h"
#include "rele/approximators/tiles/BasicTiles.h"
#include "rele/approximators/tiles/LogTiles.h"

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
    unsigned int numTiles = 6;
    Tiles* tiles[numTiles];
    tiles[0] = new BasicTiles(range, 11);
    tiles[1] = new BasicTiles(range, 10);
    tiles[2] = new BasicTiles(ranges, tilesN1);
    tiles[3] = new BasicTiles(ranges, tilesN2);
    tiles[4] = new BasicTiles(ranges, tilesN1);
    tiles[5] = new BasicTiles(ranges, tilesN2);

    for(unsigned int i = 0; i < numTiles; i++)
    {
        cout << *tiles[i] << endl;
    }

    //Inputs


    unsigned int numTest = 10000;

    cout << "## Single tiling, multidimensional Test ##" << endl;
    TilesCoder phi0(tiles[2]);

    arma::vec input = {1.55, 1.67};
    cout << "F(" << input[0] << "," << input[1] << ") = " << arma::mat(phi0(input)).t() << endl;


    cout << "## Single tiling Test ##" << endl;
    DenseTilesCoder phi1(tiles[0]);

    arma::vec phiEx1(phi1.rows(), arma::fill::zeros);
    for(unsigned int i = 0; i <= numTest; i++)
    {
        double n = numTest;
        double v = 2.0*(i)/n;
        arma::vec in = { v };

        phiEx1 += phi1(in);
    }

    std::cout << "features expectation: " << phiEx1.t() / numTest << endl;


    cout << "## Log tiles test ##" << endl;
    auto* logTiles = new LogTiles(range, 11);
    DenseTilesCoder phi2(logTiles);

    arma::vec phiEx2(phi2.rows(), arma::fill::zeros);
    for(unsigned int i = 0; i <= numTest; i++)
    {
        double n = numTest;
        double v = 2.0*(i)/n;
        arma::vec in = { v };

        phiEx2 += phi2(in);
    }

    std::cout << "features expectation: " << phiEx2.t() / numTest << endl;

    cout << "## Log tiles test ##" << endl;
    Range rangeCentered(-1.0, 1.0);
    auto* centeredlogTiles = new CenteredLogTiles(rangeCentered, 11);
    DenseTilesCoder phi3(centeredlogTiles);

    arma::vec phiEx3(phi3.rows(), arma::fill::zeros);
    for(unsigned int i = 0; i <= numTest; i++)
    {
        double n = numTest;
        double v = 2.0*i/n - 1.0;
        arma::vec in = { v };

        phiEx3 += phi3(in);
    }

    std::cout <<  "features expectation: " << phiEx3.t() / numTest << endl;


}
