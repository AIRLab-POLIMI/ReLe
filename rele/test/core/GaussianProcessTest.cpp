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

#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/regressors/others/GaussianProcess.h"
#include "rele/utils/FileManager.h"

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    cout << "## Gaussian Process regression Test ##" << endl;

    //Train plane
    BasisFunctions bfs = IdentityBasis::generate(1);
    DenseFeatures phi(bfs);

    GaussianProcess gp(phi);

    BatchDataRaw_<arma::vec, arma::vec> dataset;

    arma::vec inputs = {-7.32558,
                        -6.34884,
                        -6.34884,
                        -5.90698,
                        -4.76744,
                        -4.09302,
                        -3.74419,
                        -2.79070,
                        -2.16279,
                        -0.95349,
                        0.488372,
                        0.744186,
                        1.000000,
                        2.395349,
                        2.465116,
                        4.232558,
                        4.348837,
                        4.906977,
                        5.813953,
                        6.162791
                       };

    arma::vec outputs = {-1.753247,
                         -0.038961,
                         0.0129870,
                         0.2597402,
                         -0.818189,
                         -1.207792,
                         -1.168831,
                         0.4025974,
                         1.4285714,
                         1.7012987,
                         -0.142857,
                         -0.740260,
                         -1.000000,
                         -2.688312,
                         -2.220779,
                         -1.272727,
                         -1.077922,
                         -1.558441,
                         -1.051948,
                         -0.844156
                        };

    for(unsigned int i = 0; i < inputs.n_elem; i++)
    {
        arma::vec input = {inputs(i)};
        arma::vec output = {outputs(i)};

        dataset.addSample(input, output);
    }

    //gp.getHyperParameters().lengthScale = {0.3};
    //gp.getHyperParameters().signalVariance = 1.08;
    //gp.getHyperParameters().noiseVariance = 0.00005;

    //gp.getHyperParameters().lengthScale = {3.0};
    //gp.getHyperParameters().signalVariance = 1.16;
    //gp.getHyperParameters().noiseVariance = 0.89;

    gp.train(dataset);

    unsigned int nTestPoints = 100;
    arma::mat testInputs(1, nTestPoints, arma::fill::zeros);
    testInputs.row(0) = arma::linspace(-8, 8, nTestPoints).t();

    arma::mat testOutputs(2, nTestPoints, arma::fill::zeros);

    for(unsigned int i = 0; i < testInputs.n_cols; i++)
    {
        arma::vec results(2, arma::fill::zeros);
        results = gp(testInputs.col(i));

        cout << endl << "Input: " << testInputs(i) << endl;
        cout << "mean: " << results(0) << endl;
        cout << "variance: " << results(1) << endl;

        testOutputs.col(i) = results;
    }
    cout << endl << "marginal likelihood: " << gp.computeMarginalLikelihood() << endl;

    arma::mat trainDataset(inputs.n_elem, 2, arma::fill::zeros);
    trainDataset.col(0) = inputs;
    trainDataset.col(1) = outputs;

    arma::mat testDataset(nTestPoints, 3, arma::fill::zeros);
    testDataset.col(0) = testInputs.t();
    testDataset.col(1) = testOutputs.row(0).t();
    testDataset.col(2) = testOutputs.row(1).t();

    cout << "J: " << gp.computeJ(dataset) << endl;

    FileManager fm("GaussianProcess");
    fm.createDir();

    trainDataset.save(fm.addPath("train.mat"), arma::raw_ascii);
    testDataset.save(fm.addPath("test.mat"), arma::raw_ascii);
}
