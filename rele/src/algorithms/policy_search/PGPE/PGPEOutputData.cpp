#include "policy_search/PGPE/PGPEOutputData.h"
#include <iomanip>      // std::setprecision

using namespace std;
using namespace arma;

namespace ReLe
{

PGPEPolicyIndividual::PGPEPolicyIndividual(unsigned int nbParams, unsigned int nbEvals)
    : BlackBoxPolicyIndividual(nbParams, nbEvals), diffLogDistr(nbParams, nbEvals)
{
}

void PGPEPolicyIndividual::writeToStream(ostream& os)
{
    os << std::setprecision(9);
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
    os << diffLogDistr.n_rows << std::endl;
    for (int i = 0; i < diffLogDistr.n_cols; ++i)
    {
        for (int j = 0; j < diffLogDistr.n_rows; ++j)
        {
            os << diffLogDistr(j,i) << "\t";
        }
        os << std::endl;
    }
}

PGPEIterationStats::PGPEIterationStats(unsigned int nbIndividual,
                                       unsigned int nbParams, unsigned int nbEvals)
    : BlackBoxOutputData<PGPEPolicyIndividual>(nbIndividual, nbParams, nbEvals)
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
