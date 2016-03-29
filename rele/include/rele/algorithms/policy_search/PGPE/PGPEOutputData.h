#ifndef PGPEOUTPUTDATA_H_
#define PGPEOUTPUTDATA_H_

#include "rele/algorithms/policy_search/BlackBoxOutputData.h"
#include "rele/core/Basics.h"

namespace ReLe
{

class PGPEPolicyIndividual : public BlackBoxPolicyIndividual
{
public:
    arma::vec diffLogDistr;

public:

    PGPEPolicyIndividual(unsigned int nbParams, unsigned int nbEvals);

    virtual ~PGPEPolicyIndividual()
    {}

    void writeToStream(std::ostream& os);

    friend std::ostream& operator<<(std::ostream& out, PGPEPolicyIndividual& stat)
    {
        stat.writeToStream(out);
        return out;
    }
};

class PGPEIterationStats : public BlackBoxOutputData<PGPEPolicyIndividual>
{

public:

    PGPEIterationStats(unsigned int nbIndividual,
                       unsigned int nbParams, unsigned int nbEvals);

    virtual ~PGPEIterationStats()
    {
    }

    // AgentOutputData interface
public:
    void writeData(std::ostream& os) override;

    inline void writeDecoratedData(std::ostream& os) override
    {
        writeData(os);
    }

public:
    arma::vec metaGradient;

};


}//end namespace

#endif // PGPEOUTPUTDATA_H_
