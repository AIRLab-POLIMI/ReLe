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

#ifndef INCLUDE_RELE_ENVIRONMENTS_FOREX_H_
#define INCLUDE_RELE_ENVIRONMENTS_FOREX_H_

#include "rele/core/Basics.h"
#include "rele/core/Environment.h"

namespace ReLe
{

class Forex: public Environment<FiniteAction, FiniteState>
{
public:
    Forex(const arma::mat& rawDataset, arma::uvec whichIndicators, unsigned int actionCol);
    virtual void step(const FiniteAction& action, FiniteState& nextState,
                      Reward& reward) override;
    virtual void getInitialState(FiniteState& state) override;
    double getProfit() const;
    void setCurrentStateIdx(unsigned int currentStateIdx);
    unsigned int getNStates() const;
    const arma::mat& getDataset() const;

    virtual ~Forex();

protected:
    unsigned int getNextState(unsigned int action);
    unsigned int getStateN();

protected:
    arma::mat dataset;
    arma::vec indicatorsDim;
    double profit;
    double spread;
    unsigned int nStates;
    unsigned int currentStateIdx;
    arma::vec currentState;
    double currentPrice;
    unsigned int prevAction;
    double prevPrice;
};

}

#endif /* INCLUDE_RELE_ENVIRONMENTS_FOREX_H_ */
