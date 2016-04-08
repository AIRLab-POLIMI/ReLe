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

#include "rele/environments/Forex.h"


namespace ReLe
{

Forex::Forex(const arma::mat& rawDataset, arma::uvec whichIndicators, unsigned int priceCol) :
    dataset(arma::mat(rawDataset.n_rows - 1, whichIndicators.n_elem + 1, arma::fill::zeros)),
    indicatorsDim(arma::vec(dataset.n_cols, arma::fill::zeros)),
    profit(0),
    spread(0.0002),
    currentStateIdx(0),
    currentState(arma::vec(dataset.n_cols, arma::fill::zeros)),
    currentPrice(dataset(0, dataset.n_cols - 1)),
    prevAction(0),
    prevPrice(0)
{
    arma::uvec price = {priceCol};
    whichIndicators = arma::join_vert(whichIndicators, price);
    indicatorsDim = rawDataset.submat(arma::uvec(1, arma::fill::zeros), whichIndicators).t();

    nStates = arma::prod(indicatorsDim);

    arma::mat tempRawDataset = rawDataset.rows(arma::span(1, rawDataset.n_rows - 1));
    dataset = tempRawDataset.cols(whichIndicators);

    EnvironmentSettings& task = this->getWritableSettings();
    task.isFiniteHorizon = true;
    //task.horizon = horizon;
    task.gamma = 1;
    //task.isAverageReward = false;
    task.isEpisodic = true;
    task.statesNumber = nStates;
    task.actionsNumber = 3;
    task.stateDimensionality = 0;
    task.rewardDimensionality = 1;
}

void Forex::getInitialState(FiniteState& state)
{
    state = getNextState(0);
}

void Forex::step(const FiniteAction& action, FiniteState& nextState,
                 Reward& reward)
{
    double cost = 0;
    unsigned int currentAction = action.getActionN();

    double diff = currentPrice - prevPrice;

    if(prevAction != currentAction && currentAction != 0)
        cost = spread;

    if(prevAction == 0)
        reward[0] = -cost;
    else if(prevAction == 1)
        reward[0] = diff - cost;
    else if(prevAction == 2)
        reward[0] = -diff - cost;

    nextState = getNextState(currentAction);

    profit += reward[0];
}

unsigned int Forex::getNextState(unsigned int action)
{
    currentState(arma::span(0, currentState.n_elem - 2)) =
        dataset(currentStateIdx, arma::span(0, dataset.n_cols - 2)).t();
    currentState(currentState.n_elem - 1) = action;

    prevPrice = currentPrice;
    prevAction = action;

    currentPrice = dataset(currentStateIdx, dataset.n_cols - 1);

    currentStateIdx++;

    return getStateN();
}

unsigned int Forex::getStateN()
{
    unsigned int stateN = 0;

    unsigned int i = 0;
    for(i = 0; i < indicatorsDim.n_elem - 1; i++)
    {
        unsigned int fact = (indicatorsDim(i) == 2 ? (currentState(i) == 1 ? 0 : 1) : currentState(i));
        stateN += fact * arma::prod(indicatorsDim(arma::span(i + 1, indicatorsDim.n_elem - 1)));
    }

    unsigned int fact = (indicatorsDim(i) == 2 ? (currentState(i) == 1 ? 0 : 1) : currentState(i));
    stateN += fact;

    return stateN;
}

double Forex::getProfit() const
{
    return profit;
}

void Forex::setCurrentStateIdx(unsigned int currentStateIdx)
{
    this->currentStateIdx = currentStateIdx;
}

unsigned int Forex::getNStates() const
{
    return nStates;
}

const arma::mat& Forex::getDataset() const
{
    return dataset;
}

Forex::~Forex()
{
}

}
