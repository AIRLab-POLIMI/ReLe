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

#ifndef MULTIHEAT_H_
#define MULTIHEAT_H_

#include "DenseMDP.h"

namespace ReLe
{

class MultiHeatSettings : public EnvironmentSettings
{
public:
    MultiHeatSettings();
    static void defaultSettings(MultiHeatSettings& settings);
    virtual ~MultiHeatSettings();

public:
    unsigned int Nr;
    double Ta;
    double dt;
    double a;
    double s2n;

    arma::mat A;
    arma::vec B, C;

    double TUB, TLB;


    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);
};

class MultiHeat: public DenseMDP
{

public:

    MultiHeat();

    MultiHeat(MultiHeatSettings& config);

    virtual ~MultiHeat()
    {
        if (cleanConfig)
            delete config;
    }

    virtual void step(const FiniteAction& action, DenseState& nextState, Reward& reward);
    virtual void getInitialState(DenseState& state);

    void setCurrentState(DenseState& state) //TO REMOVE
    {
        currentState = state;
    }

private:
    unsigned int const mode = 0;

    void computeTransitionMatrix();

public:
    MultiHeatSettings* config;
    bool cleanConfig;

    arma::mat Xi;
    arma::vec Gamma;
};

}

#endif /* MULTIHEAT_H_ */
