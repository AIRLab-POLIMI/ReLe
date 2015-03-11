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

#include <rele/core/Core.h>
#include <rele/core/Agent.h>
#include "BasketBot.h"

using namespace ReLe;

class FakeAgent: public Agent<FiniteAction, FiniteState>
{
    virtual void initEpisode(const FiniteState& state, FiniteAction& action)
    {

    }

    virtual void sampleAction(const FiniteState& state, FiniteAction& action)
    {

    }
    virtual void step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action)
    {

    }

    virtual void endEpisode(const Reward& reward)
    {

    }

    virtual void endEpisode()
    {

    }
};

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "basketbot_test");
    ReLe_ROS::SimulatedBasketBot basketBot(1.0);
    FakeAgent agent;
    ReLe::Core<ReLe::FiniteAction, ReLe::FiniteState> core(basketBot, agent);

    core.getSettings().episodeLenght = 30;

    try
    {
        core.runEpisode();
        basketBot.stopSimulation();
    }
    catch (RosExitException& e)
    {
        ros::start();
        basketBot.stopSimulation();
        ros::shutdown();
        std::cout << "Node terminated" << std::endl;
    }

}
