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

#ifndef INCLUDE_RELE_UTILS_LOGGER_BATCHLOGGER_H_
#define INCLUDE_RELE_UTILS_LOGGER_BATCHLOGGER_H_

#include "rele/core/BatchCore.h"
#include "rele/core/logger/BatchLoggerStrategy.h"

namespace ReLe
{

template<class ActionC, class StateC>
class BatchLogger
{
public:
    BatchLogger() :
        strategy(nullptr)
    {
    }

    void log(Dataset<ActionC, StateC>& data)
    {
    	std::string* fileName = strategy->getFileName();
    	if(fileName)
        {
        	std::ofstream out(*fileName, std::ios_base::out);

			out << std::setprecision(OS_PRECISION);
			if(out.is_open())
				data.writeToStream(out);
			out.close();
        }
    }

    void printStatistics(AgentOutputData* outputData, unsigned int step)
    {
        if(outputData)
        {
            outputData->setStep(step);

            if(!strategy)
            {
                BatchPrintStrategy<ActionC, StateC> strategy;
                strategy.processData(outputData);
            }
            else
                strategy->processData(outputData);
        }
    }

    void setStrategy(BatchLoggerStrategy<ActionC, StateC>* strategy)
    {
        this->strategy = strategy;
    }


private:
    BatchLoggerStrategy<ActionC, StateC>* strategy;
};

}

#endif /* INCLUDE_RELE_UTILS_LOGGER_BATCHLOGGER_H_ */
