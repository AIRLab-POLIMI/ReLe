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
#include "Transition.h"

namespace ReLe
{

template<class ActionC, class StateC>
class LoggerStrategy
{
public:
    virtual void processData(Episode<ActionC, StateC>& samples) = 0;
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
    virtual void processData(Episode<ActionC, StateC>& samples)
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
    PrintStrategy(bool logTransitions = true, bool logAgent = true) :
        logTransitions(logTransitions), logAgent(logAgent)
    {

    }

    void processData(Episode<ActionC, StateC>& samples)
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
        if(logAgent)
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

    void printStateStatistics(Episode<ActionC, StateC>& samples)
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
    bool logAgent;
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


    void processData(Episode<ActionC, StateC>& samples)
    {
        if (writeTransitions)
        {
            std::ofstream ofs(transitionPath, std::ios_base::app);
            ofs << std::setprecision(OS_PRECISION);

            if(first)
            {
                samples.printHeader(ofs);
                first = false;
            }

            for(auto& sample : samples)
            {
                sample.print(ofs);
            }

            samples.back().printLast(ofs);

            ofs.close();
        }
    }

    void processData(std::vector<AgentOutputData*>& outputData)
    {
        if (writeAgentData)
        {
            std::ofstream ofs(agentDataPath, std::ios_base::app);
            ofs << std::setprecision(OS_PRECISION);

            for(auto data : outputData)
            {
                ofs << data->getStep() << "," << data->isFinal() << std::endl;
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

    void processData(Episode<ActionC, StateC>& samples)
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

template<class ActionC, class StateC>
class CollectorStrategy : public LoggerStrategy<ActionC, StateC>
{
public:
    virtual void processData(Episode<ActionC, StateC>& samples)
    {
        data.push_back(samples);
    }

    virtual void processData(std::vector<AgentOutputData*>& data)
    {
        //TODO evaluation here or abstract class...
        LoggerStrategy<ActionC, StateC>::cleanAgentOutputData(data);
    }

    virtual ~CollectorStrategy()
    {
    }

    Dataset<ActionC, StateC> data;
};

template<class ActionC, class StateC>
class MatlabCollectorStrategy : public LoggerStrategy<ActionC, StateC>
{
public:

    struct MatlabEpisode
    {
        double *states, *actions, *nextstates, *rewards;
        signed char* absorb;
        int dx,du,dr,steps;
    };

    virtual void processData(Episode<ActionC, StateC>& samples)
    {
        int ds = samples[0].x.n_elem;
        int da = samples[0].u.n_elem;
        int dr = samples[0].r.size();
        int nsteps = samples.size();
        double* states      = static_cast<double*>(malloc(ds*nsteps*sizeof(double)));
        double* nextstates  = static_cast<double*>(malloc(ds*nsteps*sizeof(double)));
        double* actions     = static_cast<double*>(malloc(da*nsteps*sizeof(double)));
        double* rewards     = static_cast<double*>(malloc(dr*nsteps*sizeof(double)));
        signed char* absorb = static_cast<signed char*>(calloc(nsteps, sizeof(signed char)));
        int count = 0;
        for (auto sample : samples)
        {
            for (int i = 0; i < ds; ++i)
            {
                states[count*ds+i] = sample.x[i];
                nextstates[count*ds+i] = sample.xn[i];
            }
            for (int i = 0; i < da; ++i)
            {
                actions[count*da+i] = sample.u[i];
            }
            for (int i = 0; i < dr; ++i)
            {
                rewards[count*dr+i] = sample.r[i];
            }
            count++;
        }
        if (samples[nsteps-1].xn.isAbsorbing())
        {
            absorb[nsteps-1] = 1;
        }

        MatlabEpisode ep;
        ep.states = states;
        ep.actions = actions;
        ep.nextstates = nextstates;
        ep.absorb = absorb;
        ep.dx = ds;
        ep.du = da;
        ep.dr = dr;
        ep.steps = nsteps;

        data.push_back(ep);
    }

    virtual void processData(std::vector<AgentOutputData*>& data)
    {
        //TODO evaluation here or abstract class...
        LoggerStrategy<ActionC, StateC>::cleanAgentOutputData(data);
    }

    virtual ~MatlabCollectorStrategy()
    {
    }

    std::vector<MatlabEpisode> data;
};

}

#endif /* INCLUDE_RELE_UTILS_LOGGERSTRATEGY_H_ */
