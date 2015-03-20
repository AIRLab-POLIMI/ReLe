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
class EmptyStrategy : public LoggerStrategy<ActionC, StateC>
{
public:
    virtual void processData(
        std::vector<Transition<ActionC, StateC>>& samples)
    {
    }

    virtual void processData(std::vector<AgentOutputData*>& outputData)
    {
        LoggerStrategy<ActionC, StateC>::cleanAgentOutputData(outputData);
    }

    virtual ~EmptyStrategy()
    {
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
                std::cout << "--- Agent data at episode end ---" << std::endl;
            }
            else
            {
                std::cout << "--- Agent data at step " << data->getStep() << " ---";
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
            std::cout << "--- Transitions ---" << std::endl;
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

    enum outType {TRANS, AGENT, ALL};

    WriteStrategy(const std::string& path, outType outputType = ALL, bool clean = false) :
        transitionPath(path), agentDataPath(addAgentOutputSuffix(path)), first(true)
    {
        if (outputType == TRANS)
        {
            writeTransitions = true;
            writeAgentData = false;
        }
        else if (outputType == AGENT)
        {
            writeAgentData = true;
            writeTransitions = false;
        }
        else
        {
            writeTransitions = true;
            writeAgentData = true;
        }
        if (clean == true)
        {
            std::ofstream os(transitionPath, std::ios_base::out);
            os.close();
            os.open(agentDataPath, std::ios_base::out);
            os.close();
        }
    }

    WriteStrategy(const std::string& transitionPath, const std::string& agentDataPath) :
        transitionPath(transitionPath), agentDataPath(agentDataPath), first(true),
        writeTransitions(true), writeAgentData(true)
    {
    }


    void processData(std::vector<Transition<ActionC, StateC>>& samples)
    {
        if (writeTransitions)
        {
            std::ofstream ofs(transitionPath, std::ios_base::app);

            if(first)
            {
                Transition<ActionC, StateC>& sample = samples[0];
                ofs << sample.x.serializedSize()  << ", "
                    << sample.u.serializedSize()  << ", "
                    << sample.r.size()  << std::endl;

                first = false;
            }

            size_t total = samples.size();
            size_t index = 0;
            for(auto& sample : samples)
            {
                index++;
                ofs << sample.x  << ", "
                    << sample.u  << ", "
                    << sample.xn << ", "
                    << sample.r  << ", "
                    << sample.xn.isAbsorbing() << ", "
                    << (index == total) << std::endl;
            }

            ofs.close();
        }
    }

    void processData(std::vector<AgentOutputData*>& outputData)
    {
        if (writeAgentData)
        {
            std::ofstream ofs(agentDataPath, std::ios_base::app);

            for(auto data : outputData)
            {
                ofs << data->getStep() << ", " << data->isFinal() << std::endl;
                data->writeData(ofs);
            }

            ofs.close();
        }

        LoggerStrategy<ActionC, StateC>::cleanAgentOutputData(outputData);
    }

private:
    std::string addAgentOutputSuffix(const std::string& path)
    {
        std::string newPath;
        size_t index = path.rfind('.');
        newPath = path.substr(0, index) + "_agentData" + path.substr(index);

        return newPath;
    }

private:
    const std::string transitionPath;
    const std::string agentDataPath;

    bool writeTransitions, writeAgentData;
    bool first;
};

template<class ActionC, class StateC>
class EvaluateStrategy : public LoggerStrategy<ActionC, StateC>
{
public:
    EvaluateStrategy(double gamma)
        : gamma(gamma)
    {
    }

    void processData(std::vector<Transition<ActionC, StateC>>& samples)
    {
        double df = 1.0;
        bool first = true;
        for (auto sample : samples)
        {
            Reward& r = sample.r;
            if (first)
            {
                J = arma::vec(r.size(), arma::fill::zeros);
                first = false;
            }
            for (int i = 0, ie = r.size(); i < ie; ++i)
            {
                J[i] += df * r[i];
            }
            df *= gamma;
        }
    }

    void processData(std::vector<AgentOutputData*>& outputData)
    {
        //TODO evaluation here or abstract class...
        LoggerStrategy<ActionC, StateC>::cleanAgentOutputData(outputData);
    }

    arma::vec J;
    double gamma;
};

}

#endif /* INCLUDE_RELE_UTILS_LOGGERSTRATEGY_H_ */
