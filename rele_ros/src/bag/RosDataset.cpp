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

#include "rele_ros/bag/RosDataset.h"

using namespace ReLe;

namespace ReLe_ROS
{

RosDataset::RosDataset(std::vector<RosTopicInterface*>& topics)
    : topics(topics)
{
    preprocessTopics();
}

void RosDataset::readEpisode(const std::string& episodePath)
{
    Episode<DenseAction, DenseState> ep;

    rosbag::Bag bag;
    bag.open(episodePath, rosbag::bagmode::Read);

    rosbag::View view(bag, rosbag::TopicQuery(topicsNames));

    DenseAction u(uDim), un(uDim);
    DenseState x(xDim), xn(xDim);

    bool first = true;
    ros::Time startTime;

    for(rosbag::MessageInstance const m : view)
    {
        for(auto* topic : topics)
        {
            arma::vec tmp;
            if(topic->readTopic(tmp, m))
            {
                unsigned int start = topic->getIndex();
                unsigned int end = start + topic->getDimension()-1;

                if(topic->isAction())
                {
                    un.rows(start, end) = tmp;
                }
                else
                {
                    xn.rows(start, end) = tmp;
                }

                if(topic->isMain())
                {
                    auto time = m.getTime();

                    if(first)
                    {
                    	startTime = time;
                    	first = false;

                    	xn(0) = 0;
                    }
                    else
                    {
                    	xn(0) = (time-startTime).toSec();

                        Transition<DenseAction, DenseState> tr;
                        tr.x = x;
                        tr.u = u;
                        tr.xn = xn; //TODO [IMPORTANT] set absorbing state
                        //tr.r = 0; //TODO [IMPORTANT] add computation of reward function

                        ep.push_back(tr);
                    }

                    x = xn;
                    u = un;

                }

                break;
            }
        }
    }

    bag.close();

    data.push_back(ep);

}

void RosDataset::preprocessTopics()
{
    xDim = 1;
    uDim = 0;

    unsigned int mainCount = 0;

    for(auto* topic : topics)
    {
        if(topic->isAction())
        {
            topic->setIndex(uDim);
            uDim += topic->getDimension();
        }
        else
        {
            topic->setIndex(xDim);
            xDim += topic->getDimension();
        }

        if(topic->isMain())
        {
        	mainCount++;
        }

        topicsNames.push_back(topic->getName());
    }

    if(mainCount < 1)
    {
    	throw std::runtime_error("At least one topic should be the main topic");
    }
}

}
