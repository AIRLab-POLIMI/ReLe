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

#include "rele/algorithms/batch/td/LSTDQ.h"
#include "rele/utils/RandomGenerator.h"

namespace ReLe
{

/**
 * Least-Squares Policy Iteration
 * Michail G. Lagoudakis and Ronald Parr
 * Journal of Machine Learning Research, 4, 2003, pp. 1107-1149.
 * Source code: https://www.cs.duke.edu/research/AI/LSPI/lspi.tar.gz
 */
template<class ActionC>
class LSPI : public BatchAgent<ActionC, DenseState>
{
public:
    LSPI(Dataset<ActionC, DenseState>& data, e_GreedyApproximate& policy,
         Features_<arma::vec>& phi, double epsilon) :
        data(data),
        oldWeights(arma::vec(phi.rows(), arma::fill::zeros)),
        policy(policy),
        phi(phi),
        critic(nullptr),
        epsilon(epsilon),
        firstStep(true)
    {
    }

    virtual void init(Dataset<ActionC, DenseState>& data, double gamma) override
    {
        critic = new LSTDQ<ActionC>(data, policy, phi, gamma);
    }

    virtual void step() override
    {
        //Evaluate the current policy (and implicitly improve)
        //            RandomGenerator::seed(1000);
        arma::vec QWeights = critic->run(firstStep);
        //            RandomGenerator::seed(1000);
        //            arma::vec Q_weights2 = critic.run_slow();
        //            arma::mat X = arma::join_horiz(Q_weights,Q_weights2);
        //            std::cout << X << std::endl;
        //            assert(max(abs(Q_weights - Q_weights2)) <=1e-3);

        critic->getQ().setParameters(QWeights);
        //            char ddd[100];
        //            sprintf(ddd,"/tmp/ReLe/w_%d.dat", iteration);
        //            Q_weights.save(ddd, arma::raw_ascii);


        firstStep = false;

        checkCond(QWeights);

        /*** Display some info ***/
        /*std::cout << "*********************************************************" << std::endl;
        if (distance > epsilon)
            std::cout << "LSPI finished in " << iteration <<
                      " iterations WITHOUT CONVERGENCE to a fixed point" << std::endl;
        else
            std::cout << "LSPI converged in " << iteration << " iterations" << std::endl;
        std::cout << "********************************************************* " << std::endl;*/
    }

    virtual void checkCond(const arma::vec& QWeights)
    {
        //Compute the distance between the current and the previous policy
        double LMAXnorm = arma::norm(QWeights - oldWeights, "inf");
        double L2norm   = arma::norm(QWeights - oldWeights, 2);
        double distance = L2norm;
        std::cout << "   Norms -> Lmax : "  << LMAXnorm <<
                  "   L2 : " << L2norm << std::endl;

        if(arma::norm(QWeights - oldWeights, 2 < epsilon))
            this->converged = true;

        oldWeights = QWeights;
    }

    virtual Policy<ActionC, DenseState>* getPolicy() override
    {
    	// FIXME
    	return nullptr;
    }

    virtual ~LSPI()
    {
        delete critic;
    }

protected:
    Dataset<ActionC, DenseState>& data;
    LSTDQ<ActionC>* critic;
    Features_<arma::vec>& phi;
    e_GreedyApproximate& policy;
    arma::vec oldWeights;
    double epsilon;
    bool firstStep;
};

}
#endif //LSPI_H_
