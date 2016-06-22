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
 * This class implements the output data for LSPI algorithm.
 */
class LSPIOutput : public AgentOutputData
{
public:
    /*!
     * Constructor.
     * \param isFinal whether the data logged comes from the end of a run of the algorithm
     * \param gamma the discount factor
     * \param QRegressor the regressor
     */
    LSPIOutput(bool isFinal, double gamma, double delta, Regressor& QRegressor);

    virtual void writeData(std::ostream& os) override;
    virtual void writeDecoratedData(std::ostream& os) override;

protected:
    double gamma;
    double delta;
    Regressor& QRegressor;
};

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

    inline virtual AgentOutputData* getAgentOutputData() override
    {
        return new LSPIOutput(false, task.gamma, delta, this->Q);
    }

    inline virtual AgentOutputData* getAgentOutputDataEnd() override
    {
        return new LSPIOutput(true, task.gamma, delta, this->Q);
    }


    virtual ~LSPI();

private:
    void checkCond(const arma::vec& QWeights);

protected:
    LSTDQ* critic;
    LinearApproximator& Q;
    arma::vec oldWeights;
    double epsilon;
    double delta;
    bool firstStep;
};

}
#endif //LSPI_H_
