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

#ifndef SWINGPENDULUM_H_
#define SWINGPENDULUM_H_

#include "DenseMDP.h"
#include "Range.h"

/**
 * Environment designed according to
 *
 * Doya, Kenji. "Reinforcement learning in continuous time and space." Neural computation 12.1 (2000): 219-245.
 * https://homes.cs.washington.edu/~todorov/courses/amath579/reading/Continuous.pdf
 *
 * Here a descrete time dynamic is used.
 */

namespace ReLe
{

class SwingUpSettings : public EnvironmentSettings
{
public:
    SwingUpSettings();
    static void defaultSettings(SwingUpSettings& settings);
    virtual ~SwingUpSettings();

public:
    double stepTime;
    double mass, length, g, requiredUpTime, upRange;
    bool useOverRotated, random_start;

    Range actionRange;
    Range thetaRange;
    Range velocityRange;
    std::vector<double> actionList;

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);
};

class DiscreteActionSwingUp: public DenseMDP
{
public:

    DiscreteActionSwingUp();
    DiscreteActionSwingUp(SwingUpSettings& config);

    void step(const FiniteAction& action, DenseState& nextState, Reward& reward);
    void getInitialState(DenseState& state);

private:
    inline void adjustTheta(double& theta)
    {
        if (theta >= M_PI)
            theta -= 2.0 * M_PI;
        if (theta < -M_PI)
            theta += 2.0 * M_PI;
    }

private:
    double previousTheta, cumulatedRotation, overRotatedTime;
    bool overRotated;
    int upTime;
    //current state [theta, velocity]
};

}//end namespace

#endif // SWINGPENDULUM_H_
