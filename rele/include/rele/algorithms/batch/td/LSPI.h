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

#include "rele/algorithms/batch/td/BatchTDAgent.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"

namespace ReLe
{

/*!
 * This class implements the Least-Squares Policy Iteration (LSPI) algorithm.
 * This algorithm is an off-policy batch algorithm that exploits
 * the action-values approximation done by the LSTDQ algorithm
 * to form an approximate policy-iteration algorithm.
 *
 * References
 * =========
 *
 * [Lagoudakis, Parr. Least-Squares Policy Iteration](http://jmlr.csail.mit.edu/papers/volume4/lagoudakis03a/lagoudakis03a.ps)
 */

class LSPI : public BatchTDAgent<DenseState>
{
private:

    class LSTDQ
    {
    public:
        LSTDQ(Dataset<FiniteAction, DenseState>& data,
              LinearApproximator& Q, double gamma, unsigned int nActions);
        arma::vec run();

    private:
        void computeDatasetFeatures();
        FiniteAction policy(const DenseState& x);


    private:
        Dataset<FiniteAction, DenseState>& data;
        LinearApproximator& Q;
        double gamma;
        unsigned int nActions;
        arma::mat Phihat;
        arma::vec Rhat;
    };

public:
    /*!
     * Constructor.
     * \param phi the features to be used for approximation
     * \param epsilon coefficient used to check whether to stop the training
     */
    LSPI(LinearApproximator& Q, double epsilon);

    virtual void init(Dataset<FiniteAction, DenseState>& data) override;
    virtual void step() override;

    virtual ~LSPI();

private:
    void checkCond(const arma::vec& QWeights);

protected:
    LSTDQ* critic;
    LinearApproximator Q;
    arma::vec oldWeights;
    double epsilon;
    bool firstStep;
};

}
#endif //LSPI_H_
