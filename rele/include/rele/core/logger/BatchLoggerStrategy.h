/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo & Matteo Pirotta
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

#ifndef INCLUDE_RELE_UTILS_LOGGER_BATCHLOGGERSTRATEGY_H_
#define INCLUDE_RELE_UTILS_LOGGER_BATCHLOGGERSTRATEGY_H_

#include <iostream>

namespace ReLe
{

template<class ActionC, class StateC>
class BatchLoggerStrategy
{
public:
    BatchLoggerStrategy(std::string* fileName) :
        fileName(fileName)
    {
    }

    virtual void processData(AgentOutputData* outputData) = 0;

    std::string* getFileName()
    {
        return fileName;
    }

    virtual ~BatchLoggerStrategy()
    {
        delete fileName;
    }


protected:
    std::string* fileName;

protected:
    void cleanAgentOutputData(AgentOutputData* outputData)
    {
        delete outputData;
    }
};

template<class ActionC, class StateC>
class BatchPrintStrategy : public BatchLoggerStrategy<ActionC, StateC>
{
public:
    BatchPrintStrategy(std::string* fileName = nullptr) :
        BatchLoggerStrategy<ActionC, StateC>(fileName)
    {
    }

    void processData(AgentOutputData* outputData) override
    {
        if(outputData->isFinal())
        {
            std::cout << std::endl << "--- Agent data at episode end ---" << std::endl;
        }
        else
        {
            std::cout << std::endl << "--- Agent data at step " << outputData->getStep() << " ---";
            std::cout << std::endl;
        }

        outputData->writeDecoratedData(std::cout);

        this->cleanAgentOutputData(outputData);
    }
};

}

#endif /* INCLUDE_RELE_UTILS_LOGGER_BATCHLOGGERSTRATEGY_H_ */
