#ifndef NESOUTPUTDATA_H_
#define NESOUTPUTDATA_H_

#include "policy_search/PGPE/PGPEOutputData.h"

namespace ReLe
{

class xNESIterationStats : public PGPEIterationStats
{

public:

    xNESIterationStats(unsigned int nbIndividual,
                       unsigned int nbParams, unsigned int nbEvals);

    virtual ~xNESIterationStats()
    {
    }

    // AgentOutputData interface
public:
    virtual void writeData(std::ostream& os);

public:
    arma::mat fisherMtx;
};

}//end namespace

#endif // NESOUTPUTDATA_H_
