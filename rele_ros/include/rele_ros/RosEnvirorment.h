/*
 * rele_ros,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef INCLUDE_ROSENVIRORMENT_H_
#define INCLUDE_ROSENVIRORMENT_H_

#include <rele/core/Envirorment.h>
#include <ros/ros.h>

namespace ReLe_ROS
{

template<class ActionC, class StateC>
class RosEnvirorment: public ReLe::Envirorment<ActionC, StateC>
{
public:
    RosEnvirorment(double controlFrequency) :
        ReLe::Envirorment(), r(controlFrequency)
    {
        setupPublishers();
        setupSubscribers();
    }

    virtual void step(const ActionC& action, StateC& nextState, ReLe::Reward& reward)
    {
        do
        {
            publishAction(action);
            ros::spinOnce();
            r.sleep();
        }
        while (stateReady);

        setState(state);
    }

    virtual void getInitialState(StateC& state)
    {
        do
        {
            ros::spinOnce();
        }
        while (stateReady);

        setState(state);
        r.reset();
    }

    virtual ~RosEnvirorment()
    {

    }

protected:
    virtual void setupPublishers() = 0;
    virtual void setupSubscribers() = 0;
    virtual void publishAction(const ActionC& action) = 0;
    virtual void setState(StateC& state) = 0;
    virtual void setReward(const ActionC& action, const StateC& state, ReLe::Reward reward) = 0;

protected:
    ros::NodeHandle n;
    ros::Rate r;
    bool stateReady;

};

}

#endif /* INCLUDE_ROSENVIRORMENT_H_ */
