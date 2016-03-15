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

#ifndef INCLUDE_RELE_ROS_ENVIRONMENTS_RRBOT_H_
#define INCLUDE_RELE_ROS_ENVIRONMENTS_RRBOT_H_

#include "rele_ros/core/SimulatedEnvironment.h"

namespace ReLe_ROS
{

class RRBot : public SimulatedEnvironment<ReLe::DenseAction,
    ReLe::DenseState>
{
public:
    RRBot(double controlFrequency);
    virtual ~RRBot();

protected:
    virtual void publishAction(const ReLe::DenseAction& action) override;
    virtual void setState(ReLe::DenseState& state) override;
    virtual void setReward(const ReLe::DenseAction& action,
                           const ReLe::DenseState& state, ReLe::Reward& reward) override;
};

}

#endif /* INCLUDE_RELE_ROS_ENVIRONMENTS_RRBOT_H_ */
