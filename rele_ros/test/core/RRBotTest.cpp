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

#include "rele_ros/environments/RRBot.h"

using namespace ReLe;

class FakeAgent: public Agent<DenseAction, DenseState>
{
    virtual void initEpisode(const DenseState& state, DenseAction& action) override
    {

    }

    virtual void sampleAction(const DenseState& state, DenseAction& action) override
    {

    }

    virtual void step(const Reward& reward, const DenseState& nextState,
    			DenseAction& action) override
    {

    }

    virtual void endEpisode(const Reward& reward) override
    {

    }

    virtual void endEpisode() override
    {

    }
};

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "rrbot_test");
    ReLe_ROS::RRBot rrbot(1.0);
    FakeAgent agent;
    auto&& core = buildCore(rrbot, agent);

    core.getSettings().episodeLength = 5;
    core.getSettings().episodeN = 3;

    try
    {
        core.runEpisodes();
    }
    catch (RosExitException& e)
    {
        std::cout << std::endl << "Node terminated by the user" << std::endl;
    }


}
