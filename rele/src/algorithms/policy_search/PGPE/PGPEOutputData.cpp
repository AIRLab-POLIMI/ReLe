#include "policy_search/PGPE/PGPEOutputData.h"

using namespace std;
using namespace arma;

namespace ReLe
{

PGPEPolicyIndividual::PGPEPolicyIndividual(unsigned int nbParams, unsigned int nbEvals)
    :Pparams(nbParams), Jvalues(nbEvals), diffLogDistr(nbParams, nbEvals)
{
}

PGPEPolicyIndividual::PGPEPolicyIndividual(arma::vec& polp, int nbEval)
    :Pparams(polp), Jvalues(nbEval), diffLogDistr(polp.n_elem, nbEval)
{
}

void PGPEPolicyIndividual::WriteToStream(ostream& os)
{
    int nparams = Pparams.n_elem;
    int nepisodes = Jvalues.n_elem;
    os << nparams;
    for (int i = 0; i < nparams; ++i)
        os << " " << Pparams[i];
    os << std::endl;
    os << nepisodes;
    for (int i = 0; i < nepisodes; ++i)
        os << " " << Jvalues[i];
    os << std::endl;
    for (int i = 0; i < nepisodes; ++i)
    {
        for (int j = 0; j < nparams; ++j)
        {
            os << diffLogDistr(j,i) << "\t";
        }
        os << std::endl;
    }
}

PGPEIterationStats::PGPEIterationStats(unsigned int nbIndividual,
                                       unsigned int nbParams, unsigned int nbEvals)
    : AgentOutputData(true)
{
    individuals.assign(nbIndividual, PGPEPolicyIndividual(nbParams, nbEvals));
}

void PGPEIterationStats::writeData(ostream &out)
{
    int i, ie = individuals.size();
    out << metaParams.n_elem << " ";
    for (i = 0; i < metaParams.n_elem; ++i)
    {
        out << metaParams[i] << " ";
    }
    out << endl;
    for (i = 0; i < metaGradient.n_elem; ++i)
    {
        out << metaGradient[i] << " ";
    }
    out << endl << ie << endl;
    for (i = 0; i < ie; ++i)
    {
        out << individuals[i];
    }
}

}