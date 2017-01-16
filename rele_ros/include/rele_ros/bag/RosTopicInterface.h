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

#ifndef INCLUDE_RELE_ROS_BAG_ROSTOPICINTERFACE_H_
#define INCLUDE_RELE_ROS_BAG_ROSTOPICINTERFACE_H_

#include <rosbag/message_instance.h>
#include <armadillo>

namespace ReLe_ROS
{

class RosTopicInterface
{
public:
    RosTopicInterface(const std::string& name, bool action, bool main);
    virtual bool readTopic(arma::vec& data, rosbag::MessageInstance const& m) = 0;
    virtual unsigned int getDimension() = 0;
    virtual ~RosTopicInterface();

    inline std::string getName() const
    {
        return topicName;
    }

    inline bool isMain() const
    {
        return main;
    }

    inline bool isAction() const
    {
        return action;
    }

    inline unsigned int getIndex() const
    {
        return index;
    }

    inline void setIndex(unsigned int index)
    {
        this->index = index;
    }

private:
    std::string topicName;
    bool action;
    bool main;
    unsigned int index;
};

template<class T>
class RosTopicInterface_ : public RosTopicInterface
{

public:
	RosTopicInterface_(const std::string& name, bool action, bool main)
		: RosTopicInterface(name, action, main)
	{

	}

    virtual bool readTopic(arma::vec& data, rosbag::MessageInstance const& m) override
    {
        typename T::ConstPtr ros_data =  m.instantiate<T>();

        if(ros_data != nullptr)
        {
            data.resize(1);
            data(0) = ros_data->data;

            return true;
        }
        else
        {
            return false;
        }
    }

    virtual unsigned int getDimension() override
    {
        return 1;
    }
};



}


#endif /* INCLUDE_RELE_ROS_BAG_ROSTOPICINTERFACE_H_ */
