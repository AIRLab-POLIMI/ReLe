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

#ifndef NLS_H_
#define NLS_H_

#include "ContinuousMDP.h"

/**
 * Environment designed according to
 *
 * Vlassis, Toussaint, Kontes, Piperidis
 * Learning Model-free Robot Control by a Monte Carlo EM Algorithm
 * Autonomous Robots 27(2):123-130, 2009
 */

namespace ReLe
{

class NLSSettings : public EnvironmentSettings
{
public:
    NLSSettings();
    static void defaultSettings(NLSSettings& settings);

public:
    double noise_mean;
    double noise_std;
    double pos0_mean;
    double pos0_std;
    double reward_reg;

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);

    virtual ~NLSSettings();
};

class NLS: public ContinuousMDP
{
public:

    NLS();
    NLS(NLSSettings& config);

    virtual ~NLS()
    {
        if (cleanConfig)
            delete config;
    }

    virtual void step(const DenseAction& action, DenseState& nextState,
                      Reward& reward) override;
    virtual void getInitialState(DenseState& state) override;

    inline const NLSSettings& getSettings() const
    {
        return *config;
    }

private:
    NLSSettings* config;
    bool cleanConfig;
};

}

#endif /* NLS_H_ */
