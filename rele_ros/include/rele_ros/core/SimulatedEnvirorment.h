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

#ifndef INCLUDE_RELE_ROS_SIMULATEDENVIRORMENT_H_
#define INCLUDE_RELE_ROS_SIMULATEDENVIRORMENT_H_

#include <stdexcept>

#include "v_repConst.h"
#include <vrep_common/simRosStartSimulation.h>
#include <vrep_common/simRosStopSimulation.h>
#include <vrep_common/simRosGetObjectHandle.h>
#include <vrep_common/simRosEnablePublisher.h>
#include <vrep_common/simRosEnableSubscriber.h>

#include <vrep_common/VrepInfo.h>
#include "RosEnvironment.h"

namespace ReLe_ROS
{

enum JointMode
{
    position = 0, targetPosition = 1, targetSpeed = 2, force_torque = 3
};

template<class ActionC, class StateC>
class SimulatedEnvirorment: public RosEnvirorment<ActionC, StateC>
{
    using RosEnvirorment<ActionC, StateC>::n;

public:
    SimulatedEnvirorment(const std::string& name, double controlFrequency) :
        RosEnvirorment<ActionC, StateC>(controlFrequency), name(name)
    {
        infoSubscriber = n.subscribe("/vrep/info", 1,
                                     &SimulatedEnvirorment::infoCallback, this);
        simulationRunning = false;
    }

    void infoCallback(const vrep_common::VrepInfo::ConstPtr& info)
    {
        /* simulationTime=info->simulationTime.data;
         simulationRunning=(info->simulatorState.data&1)!=0;
         if(!simulationRunning)
         ros::shutdown();*/
        //TODO implement this stuff
    }

    void stopSimulation()
    {
        if (simulationRunning)
        {
            stop();
        }
    }

    virtual ~SimulatedEnvirorment()
    {
    }

protected:
    virtual void start()
    {
        if (simulationRunning)
        {
            stop();
        }

        ros::service::waitForService("/vrep/simRosStartSimulation");

        vrep_common::simRosStartSimulation srv;
        ros::service::call("/vrep/simRosStartSimulation", srv);

        if (srv.response.result == -1)
        {
            throw std::runtime_error("Simulation could not be started");
        }

        simulationRunning = true;

        setupPublishSubscribe();

    }

    virtual void stop()
    {
        std::cout << "Stopping simulation" << std::endl;
        ros::service::waitForService("/vrep/simRosStopSimulation");

        vrep_common::simRosStopSimulation srv;
        ros::service::call("/vrep/simRosStopSimulation", srv);

        if (srv.response.result == -1)
        {
            throw std::runtime_error("Simulation could not be stopped");
        }

        simulationRunning = false;
        std::cout << "Simulation Stopped" << std::endl;
    }

    virtual void setupPublishSubscribe()
    {
        vrep_common::simRosEnableSubscriber srv;

        srv.request.topicName = "/" + name + "/wheels"; // the topic name
        srv.request.queueSize = 1; // the subscriber queue size (on V-REP side)
        srv.request.streamCmd = simros_strmcmd_set_joint_state; // the subscriber type

        ros::service::call("/vrep/simRosEnableSubscriber", srv);

        if (srv.response.subscriberID == -1)
        {
            throw std::runtime_error(
                "Unable to subscribe to joint state callback");
        }
    }

    bool getHandle(const std::string& name, int& handle)
    {
        vrep_common::simRosGetObjectHandle robot_handle;
        robot_handle.request.objectName = name;

        if (!ros::service::call("/vrep/simRosGetObjectHandle", robot_handle))
        {
            throw std::runtime_error("error in service call");
        }

        if (robot_handle.response.handle < 0)
        {
            throw std::runtime_error(
                "error, unable to get the handle of the vrep object "
                + name);
        }

        handle = robot_handle.response.handle;

        return true;
    }

private:
    std::string name;
    ros::Subscriber infoSubscriber;
    bool simulationRunning;
};

}
#endif /* INCLUDE_RELE_ROS_SIMULATEDENVIRORMENT_H_ */
