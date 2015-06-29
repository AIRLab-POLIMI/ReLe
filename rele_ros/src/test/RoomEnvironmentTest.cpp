/*
 * rele_ros,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirota
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
#include "RoomEnvironment.h"

using namespace ReLe;

class FakeAgent: public Agent<DenseAction, DenseState>
{
public:
    FakeAgent()
    {
        phase = 0;

        wayPoints.resize(2, arma::vec(2));

        wayPoints[0][0] = -1;
        wayPoints[0][1] = 0.5;
    }

    virtual void initEpisode(const DenseState& state, DenseAction& action)
    {
        phase = 0;

        action.resize(2);
        action[0] = 0;
        action[1] = 0;
    }
    virtual void sampleAction(const DenseState& state, DenseAction& action)
    {
        action.resize(1);
        action[0] = 2*M_PI;
        action[1] = 2*M_PI;
    }

    virtual void step(const Reward& reward, const DenseState& nextState,
                      DenseAction& action)
    {


        if(nearWaypoint(nextState, phase))
        {
            std::cout << "phase " << phase << " completed: "
                      << "[" << nextState[0] << "," << nextState[1] << "]" << std::endl;
            phase++;
        }

        action.resize(2);

        switch(phase)
        {
        case 0:
            action[0] = 2*M_PI;
            action[1] = 2*M_PI;
            break;

        case 1:
            action[0] = 0.25*2*M_PI;
            action[1] = 0.75*2*M_PI;
            break;

        default:
            std::cout << "ERROR!" << std::endl;
            break;
        }

    }

    virtual void endEpisode(const Reward& reward)
    {

    }

    virtual void endEpisode()
    {

    }

private:
    bool nearWaypoint(const DenseState& state, int i)
    {
        if(i > wayPoints.size())
            return false;

        arma::vec wp = wayPoints[i];
        arma::vec curr = state(arma::span(0,1));

        return arma::norm(curr -wp) < 0.3;
    }

private:
    int phase;

    std::vector<arma::vec> wayPoints;

};

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "room_environment_test");
    ReLe_ROS::SimulatedRoomEnvironment room(40);
    FakeAgent agent;
    Core<DenseAction, DenseState> core(room, agent);

    core.getSettings().episodeLenght = 600;

    try
    {
        core.runEpisode();
        room.stopSimulation();
    }
    catch (RosExitException& e)
    {
        ros::start();
        room.stopSimulation();
        ros::shutdown();
        std::cout << "Node terminated" << std::endl;
    }

}
