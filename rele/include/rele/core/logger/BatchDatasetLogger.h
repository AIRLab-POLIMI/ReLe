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

#ifndef INCLUDE_RELE_CORE_LOGGER_BATCHDATASETLOGGER_H_
#define INCLUDE_RELE_CORE_LOGGER_BATCHDATASETLOGGER_H_

#include <iostream>

namespace ReLe
{

template<class ActionC, class StateC>
class BatchDatasetLogger
{
public:
    virtual ~BatchDatasetLogger()
    {
    }

public:
    virtual void log(Dataset<ActionC, StateC>& data) = 0;
};

template<class ActionC, class StateC>
class CollectBatchDatasetLogger : public BatchDatasetLogger<ActionC, StateC>
{
public:
    Dataset<ActionC, StateC> data;

public:
    void log(Dataset<ActionC, StateC>& data) override
    {
        this->data = data;
    }
};

template<class ActionC, class StateC>
class WriteBatchDatasetLogger : public BatchDatasetLogger<ActionC, StateC>
{
public:
    WriteBatchDatasetLogger(std::string fileName) :
        fileName(fileName)
    {
    }

    void log(Dataset<ActionC, StateC>& data) override
    {
        std::ofstream out(fileName, std::ios_base::out);

        out << std::setprecision(OS_PRECISION);
        if(out.is_open())
            data.writeToStream(out);
        out.close();
    }

protected:
    std::string fileName;
};

}

#endif /* INCLUDE_RELE_CORE_LOGGER_BATCHDATASETLOGGER_H_ */
