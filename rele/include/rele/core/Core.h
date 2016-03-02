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

#include "rele/core/Agent.h"
#include "rele/core/Environment.h"
#include "rele/core/logger/Logger.h"

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
            episodeLength = 0;
            episodeN = 0;
            testEpisodeN = 0;
        }

        LoggerStrategy<ActionC, StateC>* loggerStrategy;
        unsigned int episodeLength;
        unsigned int episodeN;
        unsigned int testEpisodeN;

    };

public:
    Core(Environment<ActionC, StateC>& environment,
         Agent<ActionC, StateC>& agent) :
        environment(environment), agent(agent)
    {
        agent.setTask(environment.getSettings());
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
        environment.getInitialState(xn);
        agent.initEpisode(xn, u);
        logger.log(xn);

        Reward r(environment.getSettings().rewardDim);

        for (unsigned int i = 0;
                i < settings.episodeLength
                && !agent.isTerminalConditionReached(); i++)
        {

            environment.step(u, xn, r);
            logger.log(u, xn, r);

            if (xn.isAbsorbing())
            {
                agent.endEpisode(r);
                break;
            }

            agent.step(r, xn, u);
            logger.log(agent.getAgentOutputData(), i);
        }

        if (!xn.isAbsorbing())
            agent.endEpisode();

        logger.log(agent.getAgentOutputDataEnd(), settings.episodeLength);
        logger.printStatistics();
    }

    void runEpisodes()
    {
        for(unsigned int i = 0; i < settings.episodeN; i++)
        {
            runEpisode();
        }
    }

    void runTestEpisode()
    {
        //core setup
        Logger<ActionC, StateC> logger;
        StateC xn;
        ActionC u;

        logger.setStrategy(settings.loggerStrategy);

        //Start episode
        agent.initTestEpisode();
        environment.getInitialState(xn);
        logger.log(xn);

        Reward r(environment.getSettings().rewardDim);

        for (unsigned int i = 0;
                i < settings.episodeLength && !xn.isAbsorbing(); i++)
        {
            agent.sampleAction(xn, u);
            environment.step(u, xn, r);
            logger.log(u, xn, r);
        }

        logger.printStatistics();
    }

    void runTestEpisodes()
    {
        for(unsigned int i = 0; i < settings.testEpisodeN; i++)
        {
            runTestEpisode();
        }
    }

    arma::vec runBatchTest()
    {
        //core setup
        StateC xn;
        ActionC u;



        Reward r(environment.getSettings().rewardDim);
        arma::vec J_mean(r.size(), arma::fill::zeros);

        for (unsigned int e = 0; e < settings.testEpisodeN; ++e)
        {
            Logger<ActionC, StateC> logger;
            EvaluateStrategy<ActionC, StateC> stat_e(environment.getSettings().gamma);
            logger.setStrategy(&stat_e);

            //Start episode
            agent.initTestEpisode();
            environment.getInitialState(xn);
            logger.log(xn);

            for (unsigned int i = 0;
                    i < settings.episodeLength && !xn.isAbsorbing(); i++)
            {
                agent.sampleAction(xn, u);
                environment.step(u, xn, r);
                logger.log(u, xn, r);
            }

            logger.printStatistics();

            J_mean += stat_e.J;
        }

        J_mean /= settings.testEpisodeN;

        //standard deviation of J

        return J_mean;
    }

protected:
    Environment<ActionC, StateC>& environment;
    Agent<ActionC, StateC>& agent;
    CoreSettings settings;

};

template<class ActionC, class StateC>
Core<ActionC, StateC> buildCore(Environment<ActionC, StateC>& environment,
                                Agent<ActionC, StateC>& agent)
{
    return Core<ActionC, StateC>(environment, agent);
}

}

#endif /* CORE_H_ */
