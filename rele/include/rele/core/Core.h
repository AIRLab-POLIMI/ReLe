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

/*!
 * This class implements both learning and testing of an agent over an environment.
 * This class is able to run the agent on the environment, while logging both the environment and agent data.
 * Both experiment parameters and logger are configurable.
 * The core takes in account environment terminal states and agent termination conditions.
 * Also gives to the agent the environment settings, calling Agent::setTask
 */
template<class ActionC, class StateC>
class Core
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

public:
    /*!
     * This struct stores the core parameters.
     */
    struct CoreSettings
    {
        CoreSettings()
        {
            loggerStrategy = nullptr;
            episodeLength = 0;
            episodeN = 0;
            testEpisodeN = 0;
        }

        //! The logger strategy, or a null pointer if no data should be logged
        LoggerStrategy<ActionC, StateC>* loggerStrategy;
        //! The length of episodes
        unsigned int episodeLength;
        //! The number of learning episodes
        unsigned int episodeN;
        //! The number of testing episodes
        unsigned int testEpisodeN;

    };

public:
    /*!
     * Constructor.
     * \param environment the environment used for the experiment
     * \param agent the agent used for the experiment
     */
    Core(Environment<ActionC, StateC>& environment,
         Agent<ActionC, StateC>& agent) :
        environment(environment), agent(agent)
    {
        agent.setTask(environment.getSettings());
    }

    /*!
     * Getter.
     * Used to set the core parameters.
     * Example:
     * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
     * core.getSettings().loggerStrategy = new PrintStrategy<FiniteAction, FiniteState>(false);
     * core.getSettings().episodeLength = 100;
     * core.getSettings().episodeN = 1000;
     * core.getSettings().testEpisodeN = 200;
     * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     * \return a reference to the core settings.
     */
    CoreSettings& getSettings()
    {
        return settings;
    }


    /*!
     * This method runs a single learning episode.
     */
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

        Reward r(environment.getSettings().rewardDimensionality);

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

    /*!
     * This method runs the learning episodes specified in the settings.
     */
    void runEpisodes()
    {
        for(unsigned int i = 0; i < settings.episodeN; i++)
        {
            runEpisode();
        }
    }

    /*!
     * This method runs a single learning episode.
     */
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

        Reward r(environment.getSettings().rewardDimensionality);

        for (unsigned int i = 0;
                i < settings.episodeLength && !xn.isAbsorbing(); i++)
        {
            agent.sampleAction(xn, u);
            environment.step(u, xn, r);
            logger.log(u, xn, r);
        }

        logger.printStatistics();
    }

    /*!
     * This method runs the test episodes specified in the settings.
     */
    void runTestEpisodes()
    {
        for(unsigned int i = 0; i < settings.testEpisodeN; i++)
        {
            runTestEpisode();
        }
    }

    /*!
     * This method runs the test episodes specified in the settings and computes the
     * expected return of the agent w.r.t. the environment
     */
    arma::vec runEvaluation()
    {
        //core setup
        StateC xn;
        ActionC u;



        Reward r(environment.getSettings().rewardDimensionality);
        arma::vec J_mean(r.size(), arma::fill::zeros);

        //Save old logger
        auto* tmp = settings.loggerStrategy;

        //Create evaluation strategy
        EvaluateStrategy<ActionC, StateC> strategy(environment.getSettings().gamma);
        settings.loggerStrategy = &strategy;

        //Run tests
        runTestEpisodes();

        //Reset old logger
        settings.loggerStrategy = tmp;

        //return mean
        if(strategy.Jvec.size() > 0)
        {
            unsigned int rSize = strategy.Jvec[0].n_elem;
            arma::vec meanJ(rSize, arma::fill::zeros);

            for(auto& J : strategy.Jvec)
            {
                meanJ += J;
            }

            meanJ /= strategy.Jvec.size();

            return meanJ;
        }
        else
        {
            return arma::vec();
        }
    }

protected:
    Environment<ActionC, StateC>& environment;
    Agent<ActionC, StateC>& agent;
    CoreSettings settings;

};

/*!
 * This function can be used to get a core instance from an agent and an environment, reducing boilerplate code:
 * Example:
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
 * auto&& core = buildCore(environment, agent);
 * core.run();
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
template<class ActionC, class StateC>
Core<ActionC, StateC> buildCore(Environment<ActionC, StateC>& environment,
                                Agent<ActionC, StateC>& agent)
{
    return Core<ActionC, StateC>(environment, agent);
}

}

#endif /* CORE_H_ */
