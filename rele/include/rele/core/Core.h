/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
 * Versione 1.0
 *
 * This file is part of rele.
 *
 * rele is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rele is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with rele.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CORE_H_
#define CORE_H_

#include "Envirorment.h"
#include "Agent.h"
#include "logger/Logger.h"

namespace ReLe
{

template<class ActionC, class StateC>
class Core
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

public:
    struct CoreSettings
    {
        CoreSettings()
        {
            loggerStrategy = nullptr;
            episodeLenght = 0;
        }

        LoggerStrategy<ActionC, StateC>* loggerStrategy;
        unsigned int episodeLenght;

    };

public:
    Core(Envirorment<ActionC, StateC>& envirorment,
         Agent<ActionC, StateC>& agent) :
        envirorment(envirorment), agent(agent)
    {
        agent.setTask(envirorment.getSettings());
    }

    CoreSettings& getSettings()
    {
        return settings;
    }

    void runEpisode()
    {
        //core setup
        Logger<ActionC, StateC> logger;
        StateC xn;
        ActionC u;

        logger.setStrategy(settings.loggerStrategy);


        //Start episode
        envirorment.getInitialState(xn);
        agent.initEpisode(xn, u);
        logger.log(xn);

        Reward r(envirorment.getSettings().rewardDim);

        for (unsigned int i = 0;
                i < settings.episodeLenght
                && !agent.isTerminalConditionReached(); i++)
        {

            envirorment.step(u, xn, r);
            logger.log(u, xn, r);

            if (xn.isAbsorbing())
            {
                agent.endEpisode(r);
                break;
            }

            agent.step(r, xn, u);
        }

        if (!xn.isAbsorbing())
            agent.endEpisode();


        logger.printStatistics();
    }

    void runTest()
    {
        //core setup
        Logger<ActionC, StateC> logger;
        StateC xn;
        ActionC u;

        logger.setStrategy(settings.loggerStrategy);

        //Start episode
        envirorment.getInitialState(xn);
        logger.log(xn);

        Reward r(envirorment.getSettings().rewardDim);

        for (unsigned int i = 0; i < settings.episodeLenght && !xn.isAbsorbing(); i++)
        {
            agent.sampleAction(xn, u);
            envirorment.step(u, xn, r);
            logger.log(u, xn, r);
        }

        logger.printStatistics();
    }

private:
    Envirorment<ActionC, StateC>& envirorment;
    Agent<ActionC, StateC>& agent;
    CoreSettings settings;

};

}

#endif /* CORE_H_ */
