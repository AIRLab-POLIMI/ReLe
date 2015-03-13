#ifndef PGPEOUTPUTDATA_H_
#define PGPEOUTPUTDATA_H_

#include "Basics.h"

namespace ReLe
{

class PGPEPolicyIndividual
{
public:
    arma::vec Pparams;  //policy parameters
    arma::vec Jvalues;  //policy evaluation (n evaluations for each policy)
    arma::mat difflog;

public:
    PGPEPolicyIndividual(arma::vec& polp, int nbEval);

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
        stat.difflog = arma::mat(nbPolPar, nbEval);
        for (int i = 0; i < nbPolPar; ++i)
        {
            for (int j = 0; j < nbEval; ++j)
            {
                in >> stat.difflog(i,j);
            }
        }
        return in;
    }

};

class PGPEIterationStats : public AgentOutputData
{

public:

    PGPEIterationStats();

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
