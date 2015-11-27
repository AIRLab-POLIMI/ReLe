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

#ifndef INCLUDE_ROSenvironment_H_
#define INCLUDE_ROSenvironment_H_

#include <rele/core/Environment.h>
#include <ros/ros.h>

#include <iostream>
#include <exception>

class RosExitException : public std::exception
{
    //TODO... qualcosa?
};

namespace ReLe_ROS
{

template<class ActionC, class StateC>
class Rosenvironment: public ReLe::Environment<ActionC, StateC>
{
public:
    Rosenvironment(double controlFrequency) :
        r(controlFrequency)
    {
        stateReady = false;
    }

    virtual void step(const ActionC& action, StateC& nextState,
                      ReLe::Reward& reward)
    {
        do
        {
            publishAction(action);
            ros::spinOnce();
            r.sleep();
            //std::cout << "waiting for next state" << std::endl;
            checkTermination();
        }
        while (!stateReady && ros::ok());

        setState(nextState);
        setReward(action, nextState, reward);

        if (nextState.isAbsorbing())
        {
            stop();
        }
    }

    virtual void getInitialState(StateC& state)
    {
        start();

        do
        {
            ros::spinOnce();
            //std::cout << "waiting for initial state" << std::endl;
            checkTermination();
        }
        while (!stateReady && ros::ok());



        setState(state);
        r.reset();
    }

    virtual ~Rosenvironment()
    {

    }

protected:
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void publishAction(const ActionC& action) = 0;
    virtual void setState(StateC& state) = 0;
    virtual void setReward(const ActionC& action, const StateC& state,
                           ReLe::Reward& reward) = 0;

private:
    void checkTermination()
    {
        if(!ros::ok())
            throw RosExitException();
    }

protected:
    ros::NodeHandle n;
    ros::Rate r;
    bool stateReady;

};

}

#endif /* INCLUDE_ROSenvironment_H_ */
