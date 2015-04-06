#ifndef PGPEOUTPUTDATA_H_
#define PGPEOUTPUTDATA_H_

#include "policy_search/BlackBoxOutputData.h"
#include "Basics.h"

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
    void writeData(std::ostream& os);

    inline void writeDecoratedData(std::ostream& os)
    {
        writeData(os);
    }

public:
    arma::vec metaGradient;
    arma::vec stepLength;

};


}//end namespace

#endif // PGPEOUTPUTDATA_H_
