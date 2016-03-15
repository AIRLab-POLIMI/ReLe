/*
 * rele_ros,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#ifndef INCLUDE_RELE_ROS_CORE_SIMULATEDENVIRONMENT_H_
#define INCLUDE_RELE_ROS_CORE_SIMULATEDENVIRONMENT_H_

#include "RosEnvironment.h"

#include <std_srvs/Empty.h>

namespace ReLe_ROS
{

/*!
 * This class implements the basic Gazebo interface.
 * It implements the basic code for starting/stopping simulations, and
 * also print out some informations about the system.
 */
template<class ActionC, class StateC>
class SimulatedEnvironment: public RosEnvironment<ActionC, StateC>
{

public:
	/*!
	 * Constructor
	 * \param controlFrequency the frequency of the control publishing
	 */
    SimulatedEnvironment(double controlFrequency) :
        RosEnvironment<ActionC, StateC>(controlFrequency)
    {
        simulationRunning = true;

        ROS_INFO("Waiting for Gazebo");

        //wait for gazebo services
        ros::service::waitForService(reset);
        ros::service::waitForService(resetWorld);
        ros::service::waitForService(pause);
        ros::service::waitForService(resume);

        ROS_INFO("Gazebo simulator is up, stopping simulation");

        //stop simulation
        stop();
    }

    /*!
     * Destructor.
     * Stop simulation before exiting, if needed. If ros is shut down while the simulation
     * is still running, brings up the ros system again temporarily, stops the simulation,
     * and shut down the ROS system again.
     */
    virtual ~SimulatedEnvironment()
    {
        if(simulationRunning)
        {
            if(ros::ok())
            {
                ROS_INFO("Simulation is still running, stopping...");
                stop();
            }
            else
            {
                ros::start();
                stop();
                ros::shutdown();
            }
        }
    }

protected:
    /*!
     * Implementation of the start method.
     * Stops the simulation (if still running) and the starts it again.
     */
    virtual void start() override
    {
        if(simulationRunning)
            stop();

        ros::service::call(resume, emptyService);

        simulationRunning = true;
    }

    /*!
     * Implementation of the stop method.
     * Stops the simulation resetting both time and environment to the original conditions.
     */
    virtual void stop() override
    {
        ros::service::call(pause, emptyService);
        ros::service::call(reset, emptyService);
        simulationRunning = false;
    }

private:
    static constexpr auto reset = "/gazebo/reset_simulation";
    static constexpr auto resetWorld = "/gazebo/reset_world";
    static constexpr auto pause = "/gazebo/pause_physics";
    static constexpr auto resume = "/gazebo/unpause_physics";

private:
    bool simulationRunning;
    std_srvs::Empty emptyService;


};

}


#endif /* INCLUDE_RELE_ROS_CORE_SIMULATEDENVIRONMENT_H_ */
