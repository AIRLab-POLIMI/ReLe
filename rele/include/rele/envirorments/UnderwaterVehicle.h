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

#ifndef UNDERWATERVEHICLE_H_
#define UNDERWATERVEHICLE_H_

#include "DenseMDP.h"
#include "Range.h"
#include "ode45.h"

/**
 * Plant designed according to the paper
 *
 * Roland Hafner, Martin Riedmiller
 * Reinforcement learning in feedback control
 * Challenges and benchmarks from technical process control
 * Mach Learn (2011) 84:137–169
 * DOI 10.1007/s10994-011-5235-x
 */

namespace ReLe
{

class UWVSettings : public EnvirormentSettings
{
public:
    UWVSettings();
    static void defaultSettings(UWVSettings& settings);

public:
    Range thrustRange;
    Range velocityRange;
    double dt;
    double C;
    double mu; //rad
    double setPoint; // m/s
    int timeSteps;
    std::vector<double> actionList;

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);
};

class UnderwaterVehicle: public DenseMDP
{

public:
    static int ode(double t, const double x[], double dxdt[], void* params)
    {
        double* action = static_cast<double*>(params);
        const double u     = action[0];
        const double v     = x[0];
        const double abs_v = fabs(v);
        const double c_v   = 1.2f + 0.2f * sin(abs_v);
        const double m_v   = 3.0f + 1.5f * sin(abs_v);
        const double k_v_u = -0.5f * tanh((fabs(c_v * v * abs_v - u) - 30.0f) * 0.1f) + 0.5f;
        dxdt[0] = (u * k_v_u - c_v * v * abs_v) / m_v;
        return 1;
    }

public:

    UnderwaterVehicle();
    UnderwaterVehicle(UWVSettings& config);

    virtual void step(const FiniteAction& action, DenseState& nextState,
                      Reward& reward);
    virtual void getInitialState(DenseState& state);

private:
    UWVSettings uwvConfig;
    DenseState cState; //current state
};

}

#endif /* UNDERWATERVEHICLE_H_ */
