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

#include "rele_ros/environments/RoomEnvironment.h"

#include <vrep_common/JointSetStateData.h>

namespace ReLe_ROS
{

SimulatedRoomEnvironment::SimulatedRoomEnvironment(double controlFrequency)
    : SimulatedEnvironment("Roomenvironment", controlFrequency)
{

    writeSettings();

    this->getHandle("Pioneer_p3dx_leftMotor", leftMotorHandle);
    this->getHandle("Pioneer_p3dx_rightMotor", rightMotorHandle);
    this->getHandle("Pioneer_p3dx", positionHandle);

    this->getHandle("Objective", objectiveHandle);
    objective.resize(7);
    getObjectPose(objective, objectiveHandle);

    motorSpeedPub = n.advertise<vrep_common::JointSetStateData>("/Roomenvironment/wheels",1);

    stateReady = true; //FIXME LEVARE!!!
}

SimulatedRoomEnvironment::~SimulatedRoomEnvironment()
{

}

void SimulatedRoomEnvironment::publishAction(const ReLe::DenseAction& action)
{
    vrep_common::JointSetStateData motorSpeeds;

    double desiredLeftMotorSpeed = action[0];
    double desiredRightMotorSpeed = action[1];

    // publish the motor speeds:
    motorSpeeds.handles.data.push_back(leftMotorHandle);
    motorSpeeds.handles.data.push_back(rightMotorHandle);
    motorSpeeds.setModes.data.push_back(targetSpeed);
    motorSpeeds.setModes.data.push_back(targetSpeed);
    motorSpeeds.values.data.push_back(desiredLeftMotorSpeed);
    motorSpeeds.values.data.push_back(desiredRightMotorSpeed);
    motorSpeedPub.publish(motorSpeeds);

    triggerSimulation();
}

void SimulatedRoomEnvironment::setState(ReLe::DenseState& state)
{
    state.resize(7);
    while(!getObjectPose(state, positionHandle) && ros::ok());

    double distance = arma::norm(objective(arma::span(0, 2)) - state(arma::span(0, 2)));

    if(distance < 0.3)
        state.setAbsorbing();
    else
        state.setAbsorbing(false);
}

void SimulatedRoomEnvironment::setReward(const ReLe::DenseAction& action,
        const ReLe::DenseState& state, ReLe::Reward& reward)
{
    //TODO
    if(state.isAbsorbing())
        reward[0] = 0.0;
    else
        reward[0] = -1;
}


void SimulatedRoomEnvironment::writeSettings()
{
    auto& settings = this->getWritableSettings();
    settings.continuosActionDim = 2;
    settings.continuosStateDim = 1;
    settings.finiteActionDim = 0;
    settings.finiteStateDim = 0;
    settings.rewardDim = 1;
    settings.gamma = 0.9;
    settings.horizon = 0;
    settings.isFiniteHorizon = false;
    settings.isAverageReward = false;
    settings.isEpisodic = true;
}

}
