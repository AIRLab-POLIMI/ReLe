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

#include "rele/core/Core.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/environments/SwingPendulum.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/regressors/others/GaussianProcess.h"
#include "rele/utils/FileManager.h"

#include <iostream>

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{

    std::string alg = "fqi";

    unsigned int nStates = 5;

    BasisFunctions bfs = IdentityBasis::generate(nStates);
    DenseFeatures phi(bfs);

    arma::mat hParams;
    hParams.load(" ");
    arma::vec lengthScale = hParams.col(0);
    arma::vec rawSignalSigma = hParams.col(1);
    double signalSigma = arma::as_scalar(rawSignalSigma(arma::find(rawSignalSigma != arma::datum::inf)));

    if(alg == "f" || alg == "w")
    {
        std::vector<GaussianProcess> gps;

        arma::mat alpha;
        arma::cube activeSet;
        alpha.load(" ");
        activeSet.load(" ");

        for(unsigned int i = 0; i < alpha.n_cols; i++)
        {
            arma::vec rawAlphaVec = alpha.col(i);
            arma::vec alphaVec = rawAlphaVec(arma::find(rawAlphaVec != arma::datum::inf));

            GaussianProcess gp(phi);
            gp.getHyperParameters().lengthScale = lengthScale;
            gp.getHyperParameters().signalSigma = signalSigma;

            gp.setAlpha(alphaVec);

            arma::mat rawActiveSetMat = activeSet.slice(i);
            arma::vec temp = rawActiveSetMat.col(0);
            arma::mat activeSetMat = rawActiveSetMat.rows(arma::find(temp != arma::datum::inf));

            gp.setFeatures(activeSetMat);

            gps.push_back(gp);
        }
    }
    else if(alg == "d")
    {
        std::vector<GaussianProcess> gpsA;
        std::vector<GaussianProcess> gpsB;

        std::vector<std::vector<GaussianProcess>> gps;
        gps.push_back(gpsA);
        gps.push_back(gpsB);

        arma::cube alpha;
        std::vector<arma::cube> activeSets;
        alpha.load(" ");
        activeSets[0].load(" ");
        activeSets[1].load(" ");

        for(unsigned int i = 0; i < alpha.n_cols; i++)
        {
            for(unsigned int j = 0; j < gps.size(); j++)
            {
                arma::mat rawAlphaMat = alpha.slice(i);
                arma::mat rawAlphaVec = rawAlphaMat.col(0);
                arma::vec alphaVec = rawAlphaVec(arma::find(rawAlphaVec != arma::datum::inf));

                GaussianProcess gp(phi);
                gp.getHyperParameters().lengthScale = lengthScale;
                gp.getHyperParameters().signalSigma = signalSigma;

                gp.setAlpha(alphaVec);

                arma::mat rawActiveSetMat = activeSets[j].slice(i);
                arma::vec temp = rawActiveSetMat.col(0);
                arma::mat activeSetMat = rawActiveSetMat.rows(arma::find(temp != arma::datum::inf));

                gp.setFeatures(activeSetMat);

                gps[j].push_back(gp);
            }
        }
    }
}
