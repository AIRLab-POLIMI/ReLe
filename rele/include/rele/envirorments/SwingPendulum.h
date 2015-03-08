/*
 * Copyright 2014 Saminda Abeyruwan (saminda@cs.miami.edu)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SwingPendulum.h
 *
 *  Created on: Aug 25, 2012
 *      Author: sam
 */

#ifndef SWINGPENDULUM_H_
#define SWINGPENDULUM_H_

#include "DenseMDP.h"
#include "Range.h"

namespace ReLe
{

class SwingUpSettings : public EnvirormentSettings
{
public:
    SwingUpSettings();
    static void defaultSettings(SwingUpSettings& settings);

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
    SwingUpSettings config;
    double previousTheta, cumulatedRotation, overRotatedTime;
    bool overRotated;
    int upTime;
    DenseState cState; //current state [theta, velocity]
};

}//end namespace

#endif // SWINGPENDULUM_H_
