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
#include "rele/core/logger/StateStatisticGenerator.h"
#include "rele/core/Transition.h"

namespace ReLe
{

/*!
 * This class implements the basic logger strategy interface.
 * All logger strategies should implement this interface.
 */
template<class ActionC, class StateC>
class LoggerStrategy
{
public:
    /*!
     * This method describes how an episode should be processed.
     * Must be implemented.
     */
    virtual void processData(Episode<ActionC, StateC>& samples) = 0;

    /*!
     * This method describes how the agent data should be processed.
     * Must be implemented.
     */
    virtual void processData(std::vector<AgentOutputData*>& data) = 0;

    /*!
     * Destructor.
     */
    virtual ~LoggerStrategy()
    {
    }

protected:
    /*!
     * This method can be used to clean the agent data vector
     */
    void cleanAgentOutputData(std::vector<AgentOutputData*>& data)
    {
        for(auto p : data)
            delete p;
    }

};

/*!
 * This strategy can be used to print information to the console
 */
template<class ActionC, class StateC>
class PrintStrategy : public LoggerStrategy<ActionC, StateC>
{
public:
    /*!
     * Constructor.
     * \param logTransitions if the environment transitions should be printed on the console
     * \param logAgent if agent output data should be printed on the console
     */
    PrintStrategy(bool logTransitions = true, bool logAgent = true) :
        logTransitions(logTransitions), logAgent(logAgent)
    {

    }

    /*!
     * \see LoggerStrategy::processData(Episode<ActionC, StateC>& samples)
     */
    void processData(Episode<ActionC, StateC>& samples) override
    {
        printTransitions(samples);

        std::cout << std::endl << std::endl << "--- statistics ---" << std::endl
                  << std::endl;

        //print initial state
        std::cout << "- Initial State" << std::endl << "x(t0) = ["
                  << samples[0].x << "]" << std::endl;

        printStateStatistics(samples);
    }

    /*!
     * \see LoggerStrategy::processData(std::vector<AgentOutputData*>& outputData)
     */
    void processData(std::vector<AgentOutputData*>& outputData) override
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

/*!
 * This strategy can be used to save logged informations to a file.
 */
template<class ActionC, class StateC>
class WriteStrategy : public LoggerStrategy<ActionC, StateC>
{
public:
    //! enum used to select wheather to log only transitions, only agent data or both.
    enum outType {TRANS, AGENT, ALL};

    /*!
     * Constructor
     * \param path the path where to log the data
     * \param outputType what information should be logged
     * \param clean if the existing files should be overwritten up or not
     */
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

    /*!
     * Constructor
     * \param transitionPath where the transitions will be logged
     * \param agentDataPath where the agent output data will be logged
     */
    WriteStrategy(const std::string& transitionPath, const std::string& agentDataPath) :
        transitionPath(transitionPath), agentDataPath(agentDataPath), first(true),
        writeTransitions(true), writeAgentData(true)
    {
    }

    /*!
     * \see LoggerStrategy::processData(Episode<ActionC, StateC>& samples)
     */
    void processData(Episode<ActionC, StateC>& samples) override
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

    /*!
     * \see LoggerStrategy::processData(std::vector<AgentOutputData*>& outputData)
     */
    void processData(std::vector<AgentOutputData*>& outputData) override
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

/*!
 * This strategy can be used to evaluate the performances of an aget w.r.t. an environment
 */
template<class ActionC, class StateC>
class EvaluateStrategy : public LoggerStrategy<ActionC, StateC>
{
public:
    /*!
     * Constructor
     * \param gamma the discount factor for this environment
     */
    EvaluateStrategy(double gamma)
        : gamma(gamma)
    {
    }

    /*!
     * \see LoggerStrategy::processData(Episode<ActionC, StateC>& samples)
     */
    void processData(Episode<ActionC, StateC>& samples) override
    {
        double df = 1.0;
        bool first = true;

        arma::vec J(samples.getRewardSize(), arma::fill::zeros);
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

        Jvec.push_back(J);
    }

    /*!
     * \see LoggerStrategy::processData(std::vector<AgentOutputData*>& outputData)
     */
    void processData(std::vector<AgentOutputData*>& outputData) override
    {
        //TODO [MINOR][INTERFACE] evaluation here or abstract class...
        LoggerStrategy<ActionC, StateC>::cleanAgentOutputData(outputData);
    }

    //! A vector containing the returns of all episodes
    std::vector<arma::vec> Jvec;

private:
    double gamma;
};

/*!
 * This class simply collects the trajectories of all episodes into a ReLe::Dataset
 */
template<class ActionC, class StateC>
class CollectorStrategy : public LoggerStrategy<ActionC, StateC>
{
public:
    /*!
     * \see LoggerStrategy::processData(Episode<ActionC, StateC>& samples)
     */
    virtual void processData(Episode<ActionC, StateC>& samples) override
    {
        data.push_back(samples);
    }

    /*!
     * \see LoggerStrategy::processData(std::vector<AgentOutputData*>& outputData)
     */
    virtual void processData(std::vector<AgentOutputData*>& data) override
    {
        //TODO [MINOR][INTERFACE] evaluation here or abstract class...
        LoggerStrategy<ActionC, StateC>::cleanAgentOutputData(data);
    }

    /*!
     * Destructor.
     */
    virtual ~CollectorStrategy()
    {
    }

    //! the collected trajectories
    Dataset<ActionC, StateC> data;
};


inline void getDimensionsWorker(Episode<FiniteAction, FiniteState>& samples, int& ds, int& da, int& dr)
{
    ds = 1;
    da = 1;
    dr = samples[0].r.size();
}

inline void getDimensionsWorker(Episode<FiniteAction, DenseState>& samples, int& ds, int& da, int& dr)
{
    ds = samples[0].x.n_elem;
    da = 1;
    dr = samples[0].r.size();
}

inline void getDimensionsWorker(Episode<DenseAction, DenseState>& samples, int& ds, int& da, int& dr)
{
    ds = samples[0].x.n_elem;
    da = samples[0].u.n_elem;
    dr = samples[0].r.size();
}

inline void assigneActionWorker(double& val, FiniteAction& action, int i)
{
    val = action.getActionN();
}

inline void assigneActionWorker(double& val, DenseAction& action, int i)
{
    val = action[i];
}

inline void assigneStateWorker(arma::vec& val, int idx, FiniteState& state, int i)
{
    val[idx] = state.getStateN();
}

inline void assigneStateWorker(arma::vec& val, int idx, DenseState& state, int i)
{
    val[idx] = state[i];
}

}

#endif /* INCLUDE_RELE_UTILS_LOGGERSTRATEGY_H_ */
