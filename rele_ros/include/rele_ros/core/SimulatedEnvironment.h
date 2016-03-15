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

template<class ActionC, class StateC>
class SimulatedEnvironment: public RosEnvironment<ActionC, StateC>
{
protected:
    using RosEnvironment<ActionC, StateC>::n;

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
    			  ROS_INFO("Simulation is still running, stopping...");
    			  stop();
    			  ros::shutdown();
    		  }
    	  }
      }

protected:
      virtual void start() override
      {
    	  if(simulationRunning)
    		  stop();

    	  ros::service::call(resume, emptyService);

    	  simulationRunning = true;
      }

      virtual void stop() override
      {
    	  ros::service::call(pause, emptyService);
    	  ros::service::call(reset, emptyService);
    	  simulationRunning = false;
      }

      virtual void publishAction(const ActionC& action) = 0;
      virtual void setState(StateC& state) = 0;
      virtual void setReward(const ActionC& action, const StateC& state,
                             ReLe::Reward& reward) = 0;

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
