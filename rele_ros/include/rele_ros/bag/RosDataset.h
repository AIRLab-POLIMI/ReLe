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

#ifndef INCLUDE_RELE_ROS_BAG_ROSDATASET_H_
#define INCLUDE_RELE_ROS_BAG_ROSDATASET_H_

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <rele/core/Transition.h>

#include "rele_ros/bag/RosTopicInterface.h"

namespace ReLe_ROS
{

class RosDataset
{

public:
    RosDataset(std::vector<RosTopicInterface*>& topics);

    void readEpisode(const std::string& episodePath);

    inline ReLe::Dataset<ReLe::DenseAction, ReLe::DenseState>& getData()
    {
        return data;
    }

private:
    void preprocessTopics();

private:
    ReLe::Dataset<ReLe::DenseAction, ReLe::DenseState> data;
    std::vector<RosTopicInterface*> topics;

    unsigned int uDim, xDim;
    std::vector<std::string> topicsNames;
};

}

#endif /* INCLUDE_RELE_ROS_BAG_ROSDATASET_H_ */
