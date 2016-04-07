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
    //TODO [INTERFACE] qualcosa dentro l'eccezione?
};

namespace ReLe_ROS
{

/*!
 * Basic Interface for ROS environments
 * It implements the basic ros spin lop in the environments methods,
 * while offering a easy to use interface for state reading, action publishing, reward assignment.
 * This interface must be extended to implement both the low level topic publishing/subscribing
 * and the environment logic (end states, reward).
 */
template<class ActionC, class StateC>
class RosEnvironment: public ReLe::Environment<ActionC, StateC>
{
public:
    /*!
     * Constructor
     * \param controlFrequency the frequency of the control publishing
     */
    RosEnvironment(double controlFrequency) :
        r(controlFrequency)
    {
        stateReady = false;
    }

    virtual void step(const ActionC& action, StateC& nextState,
                      ReLe::Reward& reward) override
    {
        do
        {
            publishAction(action);
            ros::spinOnce();
            r.sleep();
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

    virtual void getInitialState(StateC& state) override
    {
        start();

        do
        {
            ros::spinOnce();
            checkTermination();
        }
        while (!stateReady && ros::ok());



        setState(state);
        r.reset();
    }

    virtual ~RosEnvironment()
    {

    }

protected:
    /*!
     * This method contains the logic needed to start the specific environment.
     * Must be implemented.
     */
    virtual void start() = 0;

    /*!
     * This method contains the logic needed to stop the specific environment.
     * Must be implemented.
     */
    virtual void stop() = 0;

    /*!
     * This method contains the logic used to publish actions to the system.
     * Must be implemented.
     */
    virtual void publishAction(const ActionC& action) = 0;

    /*!
     * This method contains the logic to set the current state.
     * Must be implemented.
     */
    virtual void setState(StateC& state) = 0;

    /*!
     * This method contains the reward function implementation.
     * Must be implemented.
     */
    virtual void setReward(const ActionC& action, const StateC& state,
                           ReLe::Reward& reward) = 0;

private:
    void checkTermination()
    {
        if(!ros::ok())
            throw RosExitException();
    }

protected:
    //! The node handle
    ros::NodeHandle n;

    //! The rate of the controller
    ros::Rate r;

    //! Flag to signal to if the state is ready or not, set by subclass
    bool stateReady;

};

}

#endif /* INCLUDE_ROSenvironment_H_ */
