/*
 * rele_ros,
 *
 *
 * Copyright (C) 2017 Davide Tateo
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

#ifndef INCLUDE_RELE_ROS_BAG_MESSAGE_ROSGEOMETRYINTERFACE_H_
#define INCLUDE_RELE_ROS_BAG_MESSAGE_ROSGEOMETRYINTERFACE_H_

#include "rele_ros/bag/RosTopicInterface.h"

#include <geometry_msgs/Twist.h>

namespace ReLe_ROS
{

template<>
class RosTopicInterface_<geometry_msgs::Twist> : public RosTopicInterface
{

public:

	RosTopicInterface_(const std::string& name, bool action, bool main)
		: RosTopicInterface(name, action, main)
	{

	}

    virtual bool readTopic(arma::vec& data, rosbag::MessageInstance const& m) override
    {
        typename geometry_msgs::Twist::ConstPtr ros_data =  m.instantiate<geometry_msgs::Twist>();

        if(ros_data != nullptr)
        {
            data(0) = ros_data->linear.x;
            data(1) = ros_data->linear.y;
            data(2) = ros_data->linear.z;
            data(3) = ros_data->angular.x;
            data(4) = ros_data->angular.y;
            data(5) = ros_data->angular.z;

            return true;
        }
        else
        {
            return false;
        }
    }

    virtual unsigned int getDimension() override
    {
        return 6;
    }
};

}

#endif /* INCLUDE_RELE_ROS_BAG_MESSAGE_ROSGEOMETRYINTERFACE_H_ */
