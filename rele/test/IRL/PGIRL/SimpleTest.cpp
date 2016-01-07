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

#include <armadillo>
#include <cassert>
#include "rele/utils/ArmadilloExtensions.h"

using namespace std;
using namespace ReLe;


int main()
{
    arma::mat A =
    {
        {-0.6000, 0.6000,       0},
        { 0.4000, 0.4000,       0},
        {      0,      0,  0.1000}
    };
    arma::vec weights;
    int dr = 3;
    int dp = 3;

    ////////////////////////////////////////////////
    /// PRE-PROCESSING
    ////////////////////////////////////////////////
    arma::mat Ared;         //reduced gradient matrix
    arma::uvec nonZeroIdx;  //nonzero elements of the reward weights
    int rnkG = arma::rank(A);
    if ( rnkG < dr && A.n_rows >= A.n_cols )
    {
        // select linearly independent columns
        arma::mat Asub;
        nonZeroIdx = rref(A, Asub);
        std::cout << "Asub: \n" << Asub << std::endl;
        std::cout << "idx: \n" << nonZeroIdx.t()  << std::endl;
        Ared = A.cols(nonZeroIdx);
        assert(arma::rank(Ared) == Ared.n_cols);
        //            //save idxs to be set to zero
        //            arma::vec tmp(A.n_cols);
        //            std::iota (std::begin(tmp), std::end(tmp), 0);
        //            std::vector<int> diff;
        //            std::set_difference(tmp.begin(), tmp.end(), nonZeroIdx.begin(), nonZeroIdx.end(),
        //                                   std::inserter(diff, diff.begin()));
        //            zeroIdx.set_size(diff.size());
        //            for (unsigned int i = 0, ie = diff.size(); i < ie; ++i)
        //            {
        //                zeroIdx(i) = diff[i];
        //            }
    }
    else
    {
        Ared = A;
        nonZeroIdx.set_size(A.n_cols);
        std::iota (std::begin(nonZeroIdx), std::end(nonZeroIdx), 0);
    }

    if(nonZeroIdx.n_elem == 1)
    {
        weights.zeros(A.n_cols);
        weights(nonZeroIdx).ones();

        cout << "w = " << std::endl;
        cout << weights;
        return 0;
    }


    Ared.save("/tmp/ReLe/gradRed.log", arma::raw_ascii);

    ////////////////////////////////////////////////
    /// GRAM MATRIX AND NORMAL
    ////////////////////////////////////////////////
    arma::mat gramMatrix = Ared.t() * Ared;
    //        std::cout << "Gram: \n" << gramMatrix << std::endl;
    //        arma::mat X(dr-1, dr);
    //        for (int r = 0; r < dr-1; ++r)
    //        {
    //            for (int r2 = 0; r2 < dr; ++r2)
    //            {
    //                X(r, r2) = gramMatrix(r2, r) - gramMatrix(r2, dr-1);
    //            }
    //        }
    unsigned int lastr = gramMatrix.n_rows;
    arma::mat X = gramMatrix.rows(0, lastr-2) - arma::repmat(gramMatrix.row(lastr-1), lastr-1, 1);
    //        std::cerr << std::endl << "X: " << X;
    X.save("/tmp/ReLe/GM.log", arma::raw_ascii);


    // COMPUTE NULL SPACE
    arma::mat Y = null(X);
    Y.save("/tmp/ReLe/NullS.log", arma::raw_ascii);

    cout << "w = " << std::endl;
    cout << Y;


    return 0;


}
