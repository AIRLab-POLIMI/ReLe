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

#include "rele/utils/DatasetDiscretizator.h"
#include "rele/utils/RandomGenerator.h"

using namespace ReLe;
using namespace std;
using namespace arma;

void testDenseDataset(unsigned int episodes, unsigned int transitions)
{
    Dataset<DenseAction, DenseState> dataset;

    for (unsigned int i = 0; i < episodes; i++)
    {
        Episode<DenseAction, DenseState> episode;
        bool first = true;

        arma::vec state(3, arma::fill::zeros);

        for (unsigned int j = 0; j < transitions; j++)
        {
            Transition<DenseAction, DenseState> transition;

            DenseState x(3);
            arma::vec& xr = x;
            xr = state;

            DenseAction u(2);
            arma::vec& ur = u;
            ur = arma::vec(2, arma::fill::randu);
            ur -= 0.5;
            ur(0) *= 2;

            state = arma::vec(2, arma::fill::randn);
            DenseState xn(3);
            arma::vec& xnr = xn;
            xnr = state;

            Reward r(2);
            r[0] = RandomGenerator::sampleNormal();
            r[1] = RandomGenerator::sampleUniform(-0.5, 0.5);

            transition.x = x;
            transition.u = u;
            transition.r = r;
            transition.xn = xn;

            episode.push_back(transition);
        }

        dataset.push_back(episode);
    }

    DatasetDiscretizator discretizator({Range(-1, 1), Range(-0.5, 0.5)}, {4, 2});
    auto&& datasetDiscretized = discretizator.discretize(dataset);


    dataset.writeToStream(cout);
    cout << "---------------------" << endl;
    datasetDiscretized.writeToStream(cout);
    cout << "---------------------" << endl;
}

int main(int argc, char *argv[])
{
    unsigned int episodes = 10;
    unsigned int transitions = 20;

   cout << endl << "# Testing Dense action/state dataset #" << endl << endl;
   testDenseDataset(episodes, transitions);

}
