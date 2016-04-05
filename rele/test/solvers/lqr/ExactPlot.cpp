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

#include "rele/solvers/lqr/LQRExact.h"
#include "rele/utils/FileManager.h"

using namespace ReLe;
using namespace std;

int main(int argc, char *argv[])
{
    //create folder
    FileManager fm("lqrExact");
    fm.createDir();

    //Create LQR problem
    unsigned int dim = 2;
    unsigned int rewardDim = 2;

    LQR lqr(dim, rewardDim);

    //Create lqr Exact
    LQRExact lqrExact(lqr);

    //Compute the points
    unsigned int steps = 100;

    unsigned totalSteps = std::pow(steps+1, 2);
    arma::mat J(rewardDim, totalSteps);
    arma::cube dJ(rewardDim, rewardDim, totalSteps);

    unsigned int index = 0;
    for(unsigned int i = 0; i <= steps; i++)
    {
        for(unsigned int j = 0; j <= steps; j++)
        {
            double k1 = -static_cast<double>(i)/static_cast<double>(steps);
            double k2 = -static_cast<double>(j)/static_cast<double>(steps);
            arma::vec k = {k1, k2};
            arma::mat Sigma = arma::eye(dim, dim)*0.1;

            J.col(index) = lqrExact.computeJ(k, Sigma);
            dJ.slice(index) = lqrExact.computeJacobian(k, Sigma);

            index++;
        }
    }

    std::cout << "Saving results" << std::endl;

    J.save(fm.addPath("J.txt"), arma::raw_ascii);
    dJ.save(fm.addPath("dJ.txt"), arma::raw_ascii);

    std::cout << "Done" << std::endl;
}
