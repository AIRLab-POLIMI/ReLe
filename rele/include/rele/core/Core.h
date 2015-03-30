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

#include "Agent.h"
#include "Environment.h"
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
            episodeN = 0;
            testEpisodeN = 0;
        }

        LoggerStrategy<ActionC, StateC>* loggerStrategy;
        unsigned int episodeLenght;
        unsigned int episodeN;
        unsigned int testEpisodeN;

    };

public:
    Core(Environment<ActionC, StateC>& envirorment,
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
            logger.log(agent.getAgentOutputData(), i);
        }

        if (!xn.isAbsorbing())
            agent.endEpisode();

        logger.log(agent.getAgentOutputDataEnd(), settings.episodeLenght);
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
        envirorment.getInitialState(xn);
        logger.log(xn);

        Reward r(envirorment.getSettings().rewardDim);

        for (unsigned int i = 0;
                i < settings.episodeLenght && !xn.isAbsorbing(); i++)
        {
            agent.sampleAction(xn, u);
            envirorment.step(u, xn, r);
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



        Reward r(envirorment.getSettings().rewardDim);
        arma::vec J_mean(r.size(), arma::fill::zeros);
        arma::vec J_std;

        for (unsigned int e = 0; e < settings.testEpisodeN; ++e)
        {
            Logger<ActionC, StateC> logger;
            EvaluateStrategy<ActionC, StateC> stat_e(envirorment.getSettings().gamma);
            logger.setStrategy(&stat_e);

            //Start episode
            agent.initTestEpisode();
            envirorment.getInitialState(xn);
            logger.log(xn);

            for (unsigned int i = 0;
                    i < settings.episodeLenght && !xn.isAbsorbing(); i++)
            {
                agent.sampleAction(xn, u);
                envirorment.step(u, xn, r);
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
    Environment<ActionC, StateC>& envirorment;
    Agent<ActionC, StateC>& agent;
    CoreSettings settings;

};

template<class ActionC, class StateC>
class BatchCore : protected Core<ActionC, StateC>
{

    typedef Core<ActionC, StateC> Base;
    using typename Base::CoreSettings;
    using Base::settings;
    using Base::envirorment;
    using Base::agent;

public:

    BatchCore(Environment<ActionC, StateC>& envirorment,
              Agent<ActionC, StateC>& agent,
              Dataset<ActionC, StateC>& data) :
        Core<ActionC, StateC>(envirorment, agent), data(data)
    {
        agent.setTask(envirorment.getSettings());
        nbCurrentEpisode = 0;
    }

    CoreSettings& getSettings()
    {
        return settings;
    }

    void processBatchEpisode()
    {
        //get current episode
        Episode<ActionC, StateC>& ep = data[nbCurrentEpisode];

        //core setup
        Logger<ActionC, StateC> logger;
        StateC xn;
        ActionC u;
        Reward r(envirorment.getSettings().rewardDim);

        logger.setStrategy(settings.loggerStrategy);

        //Start episode
        Transition<ActionC, StateC>& sample = ep[0];
        xn = sample.x;
        u = sample.u;
        agent.initEpisode(xn, u);
        logger.log(xn);

        int episodeLength = ep.size();

        for (unsigned int i = 0;
                i < episodeLength
                && !agent.isTerminalConditionReached(); i++)
        {
            sample = ep[i];

            //--- make a step with the model
            xn = sample.xn;
            r = sample.r;
            logger.log(u, xn, r);
            //---

            if (xn.isAbsorbing())
            {
                agent.endEpisode(r);
                break;
            }

            u = ep[i+1].u;
            agent.step(r, xn, u);
            logger.log(agent.getAgentOutputData(), i);
        }

        if (!xn.isAbsorbing())
            agent.endEpisode();

        logger.log(agent.getAgentOutputDataEnd(), episodeLength);
        logger.printStatistics();

        nbCurrentEpisode = (nbCurrentEpisode+1<data.size()) ? nbCurrentEpisode+1 : 0;
    }

    void processBatchData()
    {
        int nbEpisodes = data.size();
        nbCurrentEpisode = 0;
        for (int i = 0; i < nbEpisodes; ++i)
        {
            processBatchEpisode();
        }
    }

    arma::vec runBatchTest()
    {
        return Base::runBatchTest();
    }

private:
    Dataset<ActionC, StateC>& data;
    unsigned int nbCurrentEpisode;
};

}

#endif /* CORE_H_ */
