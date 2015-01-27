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

#include "BasketBot.h"


namespace ReLe_ROS
{

SimulatedBasketBot::SimulatedBasketBot(double controlFrequency) : SimulatedEnvirorment("basketbot", controlFrequency)
{
    stateReady = true; //FIXME LEVARE!!!
}

void SimulatedBasketBot::publishAction(const ReLe::FiniteAction& action)
{

}

void SimulatedBasketBot::setState(ReLe::FiniteState& state)
{

}

void SimulatedBasketBot::setReward(const ReLe::FiniteAction& action,
                                   const ReLe::FiniteState& state, ReLe::Reward reward)
{

}

}

