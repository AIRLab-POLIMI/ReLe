#ifndef PGPEOUTPUTDATA_H_
#define PGPEOUTPUTDATA_H_

#include "policy_search/BlackBoxOutputData.h"
#include "Basics.h"

namespace ReLe
{

class PGPEPolicyIndividual : public BlackBoxPolicyIndividual
{
public:
    arma::mat diffLogDistr;

public:

    PGPEPolicyIndividual(unsigned int nbParams, unsigned int nbEvals);

    virtual ~PGPEPolicyIndividual()
    {}

    void WriteToStream(std::ostream& os);

    friend std::ostream& operator<<(std::ostream& out, PGPEPolicyIndividual& stat)
    {
        stat.WriteToStream(out);
        return out;
    }
    friend std::istream& operator>>(std::istream& in, PGPEPolicyIndividual& stat)
    {
        int i, nbPolPar, nbEval;
        in >> nbPolPar;
        stat.Pparams = arma::vec(nbPolPar);
        for (i = 0; i < nbPolPar; ++i)
            in >> stat.Pparams[i];
        in >> nbEval;
        stat.Jvalues = arma::vec(nbEval);
        for (i = 0; i < nbEval; ++i)
            in >> stat.Jvalues[i];
        int nmetadist;
        in >> nmetadist;
        stat.diffLogDistr = arma::mat(nmetadist, nbEval);
        for (int i = 0; i < nbEval; ++i)
        {
            for (int j = 0; j < nmetadist; ++j)
            {
                in >> stat.diffLogDistr(j,i);
            }
        }
        return in;
    }

};

class PGPEIterationStats : public AgentOutputData
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

    friend std::ostream& operator<<(std::ostream& out, PGPEIterationStats& stat)
    {
        stat.writeData(out);
        return out;
    }

public:
    arma::vec metaParams;
    arma::vec metaGradient;
    std::vector<PGPEPolicyIndividual> individuals;
};

class PGPEStatistics : public std::vector<PGPEIterationStats*>
{
public:
    friend std::ostream& operator<<(std::ostream& out, PGPEStatistics& stat)
    {
        int i, ie = stat.size();
        out << ie << std::endl;
        for (i = 0; i < ie; ++i)
        {
            out << stat[i];
        }
        return out;
    }
};

}//end namespace

#endif // PGPEOUTPUTDATA_H_
