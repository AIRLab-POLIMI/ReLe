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

#ifndef INCLUDE_RELE_ROS_ENVIRORMENTS_BASKETBOT_H_
#define INCLUDE_RELE_ROS_ENVIRORMENTS_BASKETBOT_H_

#include <rele/core/Basics.h>

#include "RosEnvirorment.h"
#include "SimulatedEnvirorment.h"

namespace ReLe_ROS
{

class SimulatedBasketBot : public SimulatedEnvirorment<ReLe::FiniteAction,
    ReLe::FiniteState>
{
public:
    SimulatedBasketBot(double controlFrequency);

protected:
    virtual void publishAction(const ReLe::FiniteAction& action);
    virtual void setState(ReLe::FiniteState& state);
    virtual void setReward(const ReLe::FiniteAction& action,
                           const ReLe::FiniteState& state, ReLe::Reward reward);

private:

};

}

#endif /* INCLUDE_RELE_ROS_ENVIRORMENTS_BASKETBOT_H_ */
