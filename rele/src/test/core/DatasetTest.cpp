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

#include "Transition.h"
#include "RandomGenerator.h"

#include <sstream>

using namespace std;
using namespace ReLe;
using namespace arma;

void testFiniteDataset(unsigned int episodes, unsigned int transitions)
{
    Dataset<FiniteAction, FiniteState> dataset;

    for (unsigned int i = 0; i < episodes; i++)
    {
        Episode<FiniteAction, FiniteState> episode;
        bool first = true;
        unsigned int state = 0;
        for (unsigned int j = 0; j < transitions; j++)
        {
            Transition<FiniteAction, FiniteState> transition;
            FiniteState x(state);
            FiniteAction u(RandomGenerator::randu32());
            state = RandomGenerator::randu32();
            FiniteState xn(state);
            Reward r(1);
            r[0] = RandomGenerator::randu32();
            transition.x = x;
            transition.u = u;
            transition.r = r;
            transition.xn = xn;
            episode.push_back(transition);
        }
        dataset.push_back(episode);
    }

    stringstream ss;
    string before;
    string after;

    dataset.writeToStream(ss);
    before = ss.str();

    Dataset<FiniteAction, FiniteState> datasetReloaded;
    datasetReloaded.readFromStream(ss);

    //Clear stream
    ss.str("");
    ss.clear();

    datasetReloaded.writeToStream(ss);
    after = ss.str();

    cout << before;
    cout << "---------------------" << endl;
    cout << after;
    cout << "---------------------" << endl;

    if (before == after)
    {
        cout << "Test passed!" << endl;
        cout << "example: " << endl;
        Transition<FiniteAction, FiniteState>& t = dataset[0][0];
        Transition<FiniteAction, FiniteState>& tr = datasetReloaded[0][0];
        cout << "Dataset" << endl;
        cout << "x = " << t.x << endl;
        cout << "u = " << t.u << endl;
        cout << "r = " << t.r << endl;
        cout << "xn = " << t.xn << endl;
        cout << "Dataset reloaded" << endl;
        cout << "x = " << tr.x << endl;
        cout << "u = " << tr.u << endl;
        cout << "r = " << tr.r << endl;
        cout << "xn = " << tr.xn << endl;
    }
    else
        cout << "Test failed!" << endl;
}

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
            ur = arma::vec(2, arma::fill::randn);

            state = arma::vec(3, arma::fill::randn);
            DenseState xn(3);
            arma::vec& xnr = xn;
            xnr = state;

            Reward r(2);
            r[0] = RandomGenerator::sampleNormal();
            r[1] = RandomGenerator::sampleNormal();

            transition.x = x;
            transition.u = u;
            transition.r = r;
            transition.xn = xn;

            episode.push_back(transition);
        }

        dataset.push_back(episode);
    }

    stringstream ss;
    string before;
    string after;

    dataset.writeToStream(ss);
    before = ss.str();

    Dataset<DenseAction, DenseState> datasetReloaded;
    datasetReloaded.readFromStream(ss);

    //Clear stream
    ss.str("");
    ss.clear();

    datasetReloaded.writeToStream(ss);
    after = ss.str();

    cout << before;
    cout << "---------------------" << endl;
    cout << after;
    cout << "---------------------" << endl;

    if (before == after)
    {
        cout << "Test passed!" << endl;
        cout << "example: " << endl;
        Transition<DenseAction, DenseState>& t = dataset[0][0];
        Transition<DenseAction, DenseState>& tr = datasetReloaded[0][0];
        cout << "Dataset" << endl;
        cout << "x = " << t.x << endl;
        cout << "u = " << t.u << endl;
        cout << "r = " << t.r << endl;
        cout << "xn = " << t.xn << endl;
        cout << "Dataset reloaded" << endl;
        cout << "x = " << tr.x << endl;
        cout << "u = " << tr.u << endl;
        cout << "r = " << tr.r << endl;
        cout << "xn = " << tr.xn << endl;
    }
    else
        cout << "Test failed!" << endl;
}

void testHW()
{
    Dataset<DenseAction, DenseState> dataset;
    ifstream is("/tmp/ReLe/datahw.dat");
    dataset.readFromStream(is);
    ofstream os ("/tmp/ReLe/datahw_gen.dat");
    os << std::setprecision(OS_PRECISION);
    dataset.writeToStream(os);
    os.close();
    is.close();
}

int main(int argc, char *argv[])
{
    unsigned int episodes = 5;
    unsigned int transitions = 6;

//    cout << endl << "# Testing Finite action/state dataset #" << endl << endl;
//    testFiniteDataset(episodes, transitions);

//    cout << endl << "# Testing Dense action/state dataset #" << endl << endl;
//    testDenseDataset(episodes, transitions);

    cout << endl << "# Testing HumanWalk dataset #" << endl << endl;
    testHW();

}
