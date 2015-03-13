#include "policy_search/PGPE/PGPEOutputData.h"

using namespace std;
using namespace arma;

namespace ReLe
{

PGPEPolicyIndividual::PGPEPolicyIndividual(arma::vec &polp, int nbEval)
    :Pparams(polp), Jvalues(nbEval), difflog(polp.n_elem, nbEval)
{
}

void PGPEPolicyIndividual::WriteToStream(ostream &os)
{

}

PGPEIterationStats::PGPEIterationStats()
    : AgentOutputData(true)
{
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
