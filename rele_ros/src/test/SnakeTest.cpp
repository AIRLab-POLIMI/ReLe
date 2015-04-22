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
#include "Snake.h"

using namespace ReLe;

class FakeAgent: public Agent<DenseAction, DenseState>
{
public:
    FakeAgent()
    {

    }

    virtual void initEpisode(const DenseState& state, DenseAction& action)
    {
        arma::vec& u = action;
        u = arma::vec(8, arma::fill::randn);
    }

    virtual void sampleAction(const DenseState& state, DenseAction& action)
    {
        arma::vec& u = action;
        u = arma::vec(8, arma::fill::randn);
    }

    virtual void step(const Reward& reward, const DenseState& nextState,
                      DenseAction& action)
    {
        arma::vec& u = action;
        u = arma::vec(8, arma::fill::randn);
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
    ros::init(argc, argv, "snake_test");
    ReLe_ROS::SimulatedSnake snake(20);
    FakeAgent agent;
    Core<DenseAction, DenseState> core(snake, agent);


    PrintStrategy<DenseAction, DenseState> strategy(true, false);
    core.getSettings().episodeLenght = 100;
    core.getSettings().loggerStrategy = &strategy;

    try
    {
        core.runEpisode();
        snake.stopSimulation();
    }
    catch (RosExitException& e)
    {
        ros::start();
        snake.stopSimulation();
        ros::shutdown();
        std::cout << "Node terminated" << std::endl;
    }

}
