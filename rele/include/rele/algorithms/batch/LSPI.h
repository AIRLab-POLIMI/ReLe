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

#ifndef LSPI_H_
#define LSPI_H_

#include "batch/LSTDQ.h"

namespace ReLe
{

/**
 * Least-Squares Policy Iteration
 * Michail G. Lagoudakis and Ronald Parr
 * Journal of Machine Learning Research, 4, 2003, pp. 1107-1149.
 * Source code: https://www.cs.duke.edu/research/AI/LSPI/lspi.tar.gz
 */
template<class ActionC>
class LSPI
{
public:
    LSPI(Dataset<ActionC, DenseState>& data, e_GreedyApproximate& policy,
         Features_<arma::vec>& phi, double gamma) :
        critic(data, policy, phi, gamma)
    {
    }


    void run(unsigned int maxiterations, double epsilon = 0.1)
    {
        /*** Initialize policy iteration ***/
        unsigned int iteration = 0;
        double distance = 10 * epsilon;
        arma::vec old_weights = critic.getQ().getParameters();


        /*** Main LSPI loop ***/
        bool firsttime = true;
        while ( (iteration < maxiterations) && (distance > epsilon) )
        {
            //Update and print the number of iterations
            ++iteration;

            std::cout << "*********************************************************" << std::endl;
            std::cout << "LSPI iteration : " << iteration << std::endl;

            //Evaluate the current policy (and implicitly improve)
            arma::vec Q_weights = critic.run(firsttime);
            critic.getQ().setParameters(Q_weights);

            //Compute the distance between the current and the previous policy
            double LMAXnorm = arma::norm(Q_weights - old_weights, "inf");
            double L2norm   = arma::norm(Q_weights - old_weights, 2);
            distance = L2norm;
            std::cout << "   Norms -> Lmax : "  << LMAXnorm <<
                      "   L2 : " << L2norm << std::endl;

            firsttime = false;
            old_weights = Q_weights;
        }

        /*** Display some info ***/
        std::cout << "*********************************************************" << std::endl;
        if (distance > epsilon)
            std::cout << "LSPI finished in " << iteration <<
                      " iterations WITHOUT CONVERGENCE to a fixed point" << std::endl;
        else
            std::cout << "LSPI converged in " << iteration << " iterations" << std::endl;
        std::cout << "********************************************************* " << std::endl;
    }


protected:
    LSTDQ<ActionC> critic;

};

}
#endif //LSPI_H_
