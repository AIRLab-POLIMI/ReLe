#include "rele/algorithms/policy_search/PGPE/PGPEOutputData.h"
#include "rele/utils/CSV.h"

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
    os << std::setprecision(OS_PRECISION);
    CSVutils::vectorToCSV(Pparams, os);
    CSVutils::vectorToCSV(Jvalues, os);
    CSVutils::vectorToCSV(diffLogDistr, os);
}

PGPEIterationStats::PGPEIterationStats(unsigned int nbIndividual,
                                       unsigned int nbParams, unsigned int nbEvals)
    : BlackBoxOutputData<PGPEPolicyIndividual>(nbIndividual, nbParams, nbEvals)
{
}

void PGPEIterationStats::writeData(ostream &os)
{
    os << metaParams.n_elem << endl;
    CSVutils::vectorToCSV(metaParams, os);
    os << individuals[0].Pparams.n_elem << endl;
    os << individuals[0].Jvalues.n_elem << endl;
    os << individuals.size() << endl;
    for (auto& individual : individuals)
    {
        os << individual;
    }
    CSVutils::vectorToCSV(metaGradient, os);
}

}
