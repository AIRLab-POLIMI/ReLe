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

#ifndef PORTFOLIO_H_
#define PORTFOLIO_H_

#include "DenseMDP.h"

#define T_STEPS 50
#define N_STEPS 4
#define ALPHA 0.2
#define P_RISK 0.05
#define P_SWITCH 0.1
#define RL 1.0
#define RNL_HIGH 2
#define RNL_LOW 1.1

namespace ReLe
{

class PortfolioSettings : public EnvironmentSettings
{
public:
    PortfolioSettings();
    static void defaultSettings(PortfolioSettings& settings);
    virtual ~PortfolioSettings() {}

public:
    // time dependent variables
    unsigned int t;
    double rNL;
    double T_rNL;
    double retL;
    double retNL;
    double T_Ret_Inv;

    // time independent variables
    unsigned int T;
    unsigned int n;
    double alpha;
    double P_Risk;
    double P_Switch;
    double rL;
    double rNL_High;
    double rNL_Low;
    double initial_state;

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);
};

class Portfolio: public DenseMDP
{
public:

    Portfolio();
    Portfolio(PortfolioSettings& config);

    virtual ~Portfolio()
    {
        if (cleanConfig)
            delete config;
    }

    virtual void step(const FiniteAction& action, DenseState& nextState,
                      Reward& reward);
    virtual void getInitialState(DenseState& state);
    inline const PortfolioSettings& getSettings() const
    {
        return *config;
    }

private:
    void defaultValues();
    PortfolioSettings* config;
    bool cleanConfig;

};

}

#endif /* PORTFOLIO_H_ */
