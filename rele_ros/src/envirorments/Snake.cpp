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

#include "Snake.h"

#include <vrep_common/JointSetStateData.h>

using namespace std;

namespace ReLe_ROS
{

SimulatedSnake::SimulatedSnake(double controlFrequency, double k) :
    SimulatedEnvironment("Snake", controlFrequency), dt(1/controlFrequency), k(k)
{
    writeSettings();

    getJointsHandles();
    getBodyHandles();

    motorSpeedPub = n.advertise<vrep_common::JointSetStateData>("/Snake/wheels",1);

    stateReady = true; //FIXME LEVARE!!!
}

SimulatedSnake::~SimulatedSnake()
{

}

void SimulatedSnake::start()
{
    SimulatedEnvironment<ReLe::DenseAction, ReLe::DenseState>::start();

    //setup initial position
    arma::vec pose(7);
    while(!getObjectPose(pose, positionHandle[0]) && ros::ok());
    lastPosition = pose(arma::span(0, 2));
}


void SimulatedSnake::publishAction(const ReLe::DenseAction& action)
{
    typedef vector<float> MotorData;

    vrep_common::JointSetStateData motorSpeeds;

    motorSpeeds.handles.data = jointHandle;
    motorSpeeds.setModes.data.resize(action.size(), targetPosition);
    motorSpeeds.values.data = arma::conv_to<MotorData>::from(action);

    motorSpeedPub.publish(motorSpeeds);
    triggerSimulation();
}

void SimulatedSnake::setState(ReLe::DenseState& state)
{
    vector<arma::vec> positions(10, arma::vec(7));

    for(unsigned int i = 0; i < 10 && ros::ok(); i++)
    {
        arma::vec& position = positions[i];
        int handle = positionHandle[i];
        while(!getObjectPose(position, handle) && ros::ok());
    }

    arma::vec& x = state;

    for(auto& position : positions)
    {
        x = arma::join_vert(x, position);
    }

    state.setAbsorbing(false);
}

void SimulatedSnake::setReward(const ReLe::DenseAction& action,
                               const ReLe::DenseState& state, ReLe::Reward& reward)
{
    double oldX = lastPosition[0];
    double newX = state[0];
    double vx = (newX - oldX) / dt;

    const arma::vec& u = action;
    reward[0] = vx - k*arma::norm(u);

    lastPosition = state(arma::span(0, 2));
}

void SimulatedSnake::writeSettings()
{
    auto& settings = this->getWritableSettings();
    settings.continuosActionDim = 8;
    settings.continuosStateDim = 70;
    settings.finiteActionDim = 0;
    settings.finiteStateDim = 0;
    settings.rewardDim = 1;
    settings.gamma = 0.9;
    settings.horizon = 0;
    settings.isFiniteHorizon = false;
    settings.isAverageReward = false;
    settings.isEpisodic = true;
}


void SimulatedSnake::getJointsHandles()
{
    jointHandle.resize(8);
    for (unsigned int i = 0; i < 4; i++)
    {
        string nameV = "snake_joint_v" + to_string(i + 1);
        string nameH = "snake_joint_h" + to_string(i + 1);
        int& handleV = jointHandle[2*i];
        int& handleH = jointHandle[2*i+1];
        this->getHandle(nameV, handleV);
        this->getHandle(nameH, handleH);
    }
}

void SimulatedSnake::getBodyHandles()
{
    positionHandle.resize(10);

    for (unsigned int i = 0; i < 10; i++)
    {
        string nameBody = "snake_body" + to_string(i + 1);
        int& handleBody = positionHandle[i];
        this->getHandle(nameBody, handleBody);
    }
}

}
