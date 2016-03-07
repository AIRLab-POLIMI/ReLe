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

#ifndef INCLUDE_RELE_UTILS_LOGGER_BATCHAGENTLOGGER_H_
#define INCLUDE_RELE_UTILS_LOGGER_BATCHAGENTLOGGER_H_

#include "rele/core/BatchCore.h"

namespace ReLe
{

template<class ActionC, class StateC>
class BatchAgentLogger
{
public:
    void log(AgentOutputData* outputData, unsigned int step)
    {
        if(outputData)
        {
            outputData->setStep(step);
			processData(outputData);
        }
    }

    virtual ~BatchAgentLogger()
    {
    }

protected:
	virtual void processData(AgentOutputData* outputData) = 0;
};

template<class ActionC, class StateC>
class BatchAgentPrintLogger : public BatchAgentLogger<ActionC, StateC>
{
protected:
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

        delete outputData;
    }
};

}

#endif /* INCLUDE_RELE_UTILS_LOGGER_BATCHAGENTLOGGER_H_ */
