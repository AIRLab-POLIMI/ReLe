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

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_BATCHAGENT_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_BATCHAGENT_H_

#include "rele/approximators/data/BatchData.h"

namespace ReLe
{

template<class ActionC, class StateC>
class BatchAgent
{
public:
<<<<<<< Updated upstream
    BatchAgent(double gamma, unsigned int nMiniBatches = 1) :
        nMiniBatches(nMiniBatches),
        gamma(gamma),
        terminalCond(nullptr)
    {
    }
=======
	BatchAgent(double gamma) :
		gamma(gamma),
		converged(false)
	{
	}
>>>>>>> Stashed changes

    virtual void init(Dataset<FiniteAction, StateC>& data) = 0;
    virtual void step() = 0;

    virtual AgentOutputData* getAgentOutputData()
    {
        return nullptr;
    }

    virtual bool isConverged()
    {
    	return converged;
    }

    virtual ~BatchAgent()
    {
    }

protected:
<<<<<<< Updated upstream
    unsigned int nMiniBatches;
    double gamma;
    TerminalCondition* terminalCond;
=======
	double gamma;
    bool converged;
>>>>>>> Stashed changes
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_BATCHAGENT_H_ */
