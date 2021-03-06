#ifndef NESOUTPUTDATA_H_
#define NESOUTPUTDATA_H_

#include "rele/algorithms/policy_search/PGPE/PGPEOutputData.h"

namespace ReLe
{

class NESIterationStats : public PGPEIterationStats
{

public:

    NESIterationStats(unsigned int nbIndividual,
                      unsigned int nbParams, unsigned int nbEvals);

    virtual ~NESIterationStats()
    {
    }

    // AgentOutputData interface
public:
    virtual void writeData(std::ostream& os) override;

public:
    arma::mat fisherMtx;
};

}//end namespace

#endif // NESOUTPUTDATA_H_
