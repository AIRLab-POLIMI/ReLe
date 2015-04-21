/*
 * rele_ros,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
 * Versione 1.0
 *
 * This file is part of rele_ros.
 *
 * rele_ros is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rele_ros is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with rele_ros.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef INCLUDE_RELE_ROS_ENVIRORMENTS_SNAKE_H_
#define INCLUDE_RELE_ROS_ENVIRORMENTS_SNAKE_H_

#include "../core/SimulatedEnvironment.h"

namespace ReLe_ROS
{

class SimulatedSnake : public SimulatedEnvironment<ReLe::DenseAction, ReLe::DenseState>
{
public:
    SimulatedSnake(double controlFrequency);

protected:
    virtual void publishAction(const ReLe::DenseAction& action);
    virtual void setState(ReLe::DenseState& state);
    virtual void setReward(const ReLe::DenseAction& action,
                           const ReLe::DenseState& state, ReLe::Reward& reward);

private:
    void writeSettings();
    void getJointsHandles();
    void getBodyHandles();

private:
    std::vector<int> jointHandle;
    std::vector<int> vJointHandle;

    std::vector<int> positionHandle;

    ros::Publisher motorSpeedPub;
};


}


#endif /* INCLUDE_RELE_ROS_ENVIRORMENTS_SNAKE_H_ */
