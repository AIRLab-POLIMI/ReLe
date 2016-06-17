/*
*  rele,
*
*
*  Copyright (C) 2015 Davide Tateo & Matteo Pirotta
*  Versione 1.0
*
*  This file is part of rele.
*
*  rele is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  rele is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with rele.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "rele/algorithms/batch/td/FQI.h"

namespace ReLe
{

FQIOutput::FQIOutput(bool isFinal, double gamma, double delta, Regressor& QRegressor) :
    AgentOutputData(isFinal),
    gamma(gamma),
    delta(delta),
    QRegressor(QRegressor)
{
}

void FQIOutput::writeData(std::ostream& os)
{
    os << "delta: " << delta << std::endl;
    os << "- Parameters" << std::endl;
    os << "gamma: " << gamma << std::endl;
}

void FQIOutput::writeDecoratedData(std::ostream& os)
{
    os << "delta: " << delta << std::endl;
    os << "- Parameters" << std::endl;
    os << "gamma: " << gamma << std::endl;
}

FQI::FQI(BatchRegressor& QRegressor, double epsilon) :
    BatchTDAgent<DenseState>(QRegressor),
    Q(QRegressor),
    nSamples(0),
    firstStep(true),
    epsilon(epsilon),
    delta(std::numeric_limits<double>::infinity())
{
}

void FQI::init(Dataset<FiniteAction, DenseState>& data)
{
    features = data.featuresAsMatrix(Q.getFeatures());
    nSamples = features.n_cols;
    states = arma::mat(task.stateDimensionality,
                       nSamples,
                       arma::fill::zeros);
    actions = arma::vec(nSamples, arma::fill::zeros);
    nextStates = arma::mat(task.stateDimensionality,
                           nSamples,
                           arma::fill::zeros);
    rewards = arma::vec(nSamples, arma::fill::zeros);
    QHat = arma::vec(nSamples, arma::fill::zeros);

    unsigned int i = 0;
    for(auto& episode : data)
        for(auto& tr : episode)
        {
            if(tr.xn.isAbsorbing())
                absorbingStates.insert(i);

            states.col(i) = tr.x;
            actions(i) = tr.u;
            nextStates.col(i) = tr.xn;
            rewards(i) = tr.r[0];
            i++;
        }
}

void FQI::step()
{
    arma::mat outputs(1, nSamples, arma::fill::zeros);

    for(unsigned int i = 0; i < nSamples; i++)
    {
        arma::vec Q_xn(task.actionsNumber, arma::fill::zeros);
        if(absorbingStates.count(i) == 0 && !firstStep)
        {
            for(unsigned int u = 0; u < task.actionsNumber; u++)
                Q_xn(u) = arma::as_scalar(this->Q(nextStates.col(i),
                                                  FiniteAction(u)));

            outputs(i) = rewards(i) + task.gamma * arma::max(Q_xn);
        }
        else
            outputs(i) = rewards(i);
    }

    BatchDataSimple featureDataset(features, outputs);
    this->Q.trainFeatures(featureDataset);

    firstStep = false;

    checkCond();
}

void FQI::checkCond()
{
    arma::vec prevQHat = QHat;

    computeQHat();

    delta = arma::norm(QHat - prevQHat);

    if(delta < epsilon)
        this->converged = true;
}

void FQI::computeQHat()
{
    for(unsigned int i = 0; i < states.n_cols; i++)
        QHat(i) = arma::as_scalar(Q(states.col(i), FiniteAction(actions(i)))(0));
}

}
