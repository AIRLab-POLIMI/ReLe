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

#ifndef INCLUDE_RELE_UTILS_LOGGERSTRATEGY_H_
#define INCLUDE_RELE_UTILS_LOGGERSTRATEGY_H_

#include <iostream>
#include "StateStatisticGenerator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class LoggerStrategy
{
public:
    virtual void processData(
        std::vector<Transition<ActionC, StateC>>& samples) = 0;
    virtual void processData(std::vector<AgentOutputData*>& data) = 0;

    virtual ~LoggerStrategy()
    {
    }

protected:
    void cleanAgentOutputData(std::vector<AgentOutputData*>& data)
    {
        for(auto p : data)
            delete p;
    }

};

template<class ActionC, class StateC>
class PrintStrategy : public LoggerStrategy<ActionC, StateC>
{
public:
    PrintStrategy(bool logTransitions = true) :
        logTransitions(logTransitions)
    {

    }

    void processData(std::vector<Transition<ActionC, StateC>>& samples)
    {
        printTransitions(samples);

        std::cout << std::endl << std::endl << "--- statistics ---" << std::endl
                  << std::endl;

        //print initial state
        std::cout << "- Initial State" << std::endl << "x(t0) = ["
                  << samples[0].x << "]" << std::endl;

        printStateStatistics(samples);
    }

    void processData(std::vector<AgentOutputData*>& outputData)
    {
        for(auto data : outputData)
        {
            if(data->isFinal())
            {
                std::cout << "- Agent data at episode end" << std::endl;
            }
            else
            {
                std::cout << "- Agent data at step " << data->getStep();
                std::cout << std::endl;
            }

            data->writeDecoratedData(std::cout);
        }

        LoggerStrategy<ActionC, StateC>::cleanAgentOutputData(outputData);
    }

private:
    void printTransitions(std::vector<Transition<ActionC, StateC>>& samples)
    {
        if (logTransitions)
        {
            std::cout << "- Transitions" << std::endl;
            int t = 0;
            for (auto sample : samples)
            {
                auto& x = sample.x;
                auto& u = sample.u;
                auto& xn = sample.xn;
                Reward& r = sample.r;
                std::cout << "t = " << t++ << ": x = [" << x << "] u = [" << u
                          << "] xn = [" << xn << "] r = [" << r << "]"
                          << std::endl;
            }
        }
    }

    void printStateStatistics(std::vector<Transition<ActionC, StateC>>& samples)
    {
        std::cout << "- State Statistics" << std::endl;

        for(auto& transition : samples)
        {
            stateStatisticsGenerator.addStateVisit(transition.xn);
        }

        std::cout << stateStatisticsGenerator.to_str() << std::endl;

    }

private:
    bool logTransitions;
    StateStatisticGenerator<StateC> stateStatisticsGenerator;

};

template<class ActionC, class StateC>
class WriteStrategy : public LoggerStrategy<ActionC, StateC>
{
public:
    WriteStrategy(const std::string& path) :
        transitionPath(path), agentDataPath(path)
    {

    }

    WriteStrategy(const std::string& transitionPath, const std::string& agentDataPath) :
        transitionPath(transitionPath), agentDataPath(agentDataPath)
    {

    }

    void processData(std::vector<Transition<ActionC, StateC>>& samples)
    {
        std::ofstream ofs(transitionPath); //TODO append?
        //TODO print data as matrix
        ofs.close();
    }

    void processData(std::vector<AgentOutputData*>& outputData)
    {
        std::ofstream ofs(agentDataPath); //TODO append?
        //TODO print data as matrix
        ofs.close();

        LoggerStrategy<ActionC, StateC>::cleanAgentOutputData(outputData);
    }

private:
    const std::string& transitionPath;
    const std::string& agentDataPath;
};

template<class ActionC, class StateC>
class EvaluateStrategy : public LoggerStrategy<ActionC, StateC>
{
public:
    EvaluateStrategy()
    {

    }

    void processData(std::vector<Transition<ActionC, StateC>>& samples)
    {
        //TODO evaluation here or abstract class...
        int t = 0;
        for (auto sample : samples)
        {
            Reward& r = sample.r;
            if (t == 0)
            {
                J_mean = arma::vec(r.size(), arma::fill::zeros);
                J_standarDeviation = arma::vec(r.size(), arma::fill::zeros);
            }
            for (int i = 0, ie = r.size(); i < ie; ++i)
            {
                J_mean[i] += r[i];
            }
            ++t;
        }
        for (int i = 0, ie = J_mean.n_elem; i < ie; ++i)
            J_mean[i] /= t;

        for (auto sample : samples)
        {
            Reward& r = sample.r;
            for (int i = 0, ie = r.size(); i < ie; ++i)
            {
                double v = r[i] - J_mean[i];
                J_standarDeviation[i] += v*v;
            }
        }

        for (int i = 0, ie = J_standarDeviation.n_elem; i < ie; ++i)
            J_standarDeviation[i] /= t;
    }

    void processData(std::vector<AgentOutputData*>& outputData)
    {
        //TODO evaluation here or abstract class...
        LoggerStrategy<ActionC, StateC>::cleanAgentOutputData(outputData);
    }

    arma::vec J_mean;
    arma::vec J_standarDeviation;
};

}

#endif /* INCLUDE_RELE_UTILS_LOGGERSTRATEGY_H_ */
