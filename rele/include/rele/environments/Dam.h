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

#ifndef DAM_H_
#define DAM_H_

#include "ContinuousMDP.h"

namespace ReLe
{

class DamSettings : public EnvirormentSettings
{
public:
    DamSettings();
    static void defaultSettings(DamSettings& settings);
    virtual ~DamSettings() {}

public:
    double S;                      // reservoir surface
    double W_IRR;                 // water demand
    double H_FLO_U;               // flooding threshold
    double S_MIN_REL;
    double DAM_INFLOW_MEAN;
    double DAM_INFLOW_STD;
    double Q_MEF;
    double GAMMA_H2O;
    double W_HYD;                 //  hydroelectric demand
    double Q_FLO_D;
    double ETA;
    double G;

    std::vector<double> normalization_factor;

    bool penalize;

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);
};

class Dam: public ContinuousMDP
{
public:

    Dam();
    Dam(DamSettings& config);

    virtual void step(const DenseAction& action, DenseState& nextState,
                      Reward& reward);
    virtual void getInitialState(DenseState& state);

private:
    DamSettings damConfig;
};

}

#endif /* DAM_H_ */
