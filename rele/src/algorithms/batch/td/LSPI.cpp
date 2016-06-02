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

#include "rele/algorithms/batch/td/LSPI.h"

namespace ReLe
{


LSPI::LSTDQ::LSTDQ(Dataset<FiniteAction, DenseState>& data,
                   LinearApproximator& Q, double gamma, unsigned int nActions)
    : data(data), Q(Q), gamma(gamma), nActions(nActions)
{
    computeDatasetFeatures();
}

arma::vec LSPI::LSTDQ::run()
{
    // Initialize variables
    int nbEpisodes = this->data.size();
    int nbSamples = data.getTransitionsNumber();

    Features& phi = Q.getFeatures();
    int df = phi.rows();

    arma::mat PiPhihat(nbSamples, df, arma::fill::zeros);

    unsigned int idx = 0;
    for (auto episode : this->data)
    {
        for (auto tr : episode)
        {
            auto nextAction = policy(tr.xn);
            arma::mat nextPhi = phi(tr.xn, nextAction);
            PiPhihat.row(idx) = nextPhi.t();

            // increment sample counter
            ++idx;
        }
    }

    // Compute the matrices A and b
    arma::mat A = Phihat.t() * (Phihat - gamma * PiPhihat);
    arma::vec b = Phihat.t() * Rhat;

    // Solve the system to find w
    arma::vec w;
    int rank = arma::rank(A);
    if (rank == df)
    {
        w = arma::solve(A,b);
    }
    else
    {
        std::cout << "Matrix is not invertible" << std::endl;
        w = arma::pinv(A)*b;
    }

    return w;
}

void LSPI::LSTDQ::computeDatasetFeatures()
{
    // Initialize variables
    int nbEpisodes = this->data.size();
    int nbSamples = data.getTransitionsNumber();

    Features& phi = Q.getFeatures();
    int df = phi.rows();


    // Precompute Phihat and Rhat for all subsequent iterations
    Phihat.set_size(nbSamples,df);
    Rhat.set_size(nbSamples);

    unsigned int idx = 0;
    for (auto episode : this->data)
    {
        for (auto tr : episode)
        {
            //compute basis in current state-action pair
            arma::vec phi_xu = phi(tr.x, tr.u);

            //update matricies
            Phihat.row(idx) = phi_xu.t();
            Rhat(idx) = tr.r[0];

            // increment sample counter
            ++idx;
        }
    }
}

FiniteAction LSPI::LSTDQ::policy(const DenseState& x)
{
    Features& phi = Q.getFeatures();
    Regressor& Q = this->Q;
    arma::vec qValues(nActions, arma::fill::zeros);
    for(unsigned int i = 0; i < nActions; i++)
    {
        FiniteAction u(i);
        qValues(u) = arma::as_scalar(Q(x, u));
    }

    arma::uword maxQIndex;
    qValues.max(maxQIndex);

    FiniteAction uMax(maxQIndex);

    return uMax;
}

LSPI::LSPI(LinearApproximator& Q, double epsilon) :
    BatchTDAgent<DenseState>(Q),
    oldWeights(Q.getParametersSize(), arma::fill::zeros),
    Q(Q),
    critic(nullptr),
    epsilon(epsilon),
    firstStep(true)
{
}

void LSPI::init(Dataset<FiniteAction, DenseState>& data)
{
    critic = new LSTDQ(data, Q, task.gamma, task.actionsNumber);
    firstStep = true;
    this->converged = false;
}

void LSPI::step()
{
    //Evaluate the current policy (and implicitly improve)
    arma::vec QWeights = critic->run();
    Q.setParameters(QWeights);

    //check if termination condition has been reached
    if(!firstStep)
        checkCond(QWeights);

    //save old weights
    oldWeights = QWeights;

    //set first step variable
    firstStep = false;
}

void LSPI::checkCond(const arma::vec& QWeights)
{
    //Compute the distance between the current and the previous policy
    double distance = arma::norm(QWeights - oldWeights, 2);

    if(distance < epsilon)
        this->converged = true;

    static int i = 0;
    std::cout << i++ << " " << distance << std::endl;
}

LSPI::~LSPI()
{
    delete critic;
}


}
