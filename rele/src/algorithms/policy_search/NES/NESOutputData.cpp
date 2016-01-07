#include "rele/algorithms/policy_search/NES/NESOutputData.h"
#include "rele/utils/CSV.h"

using namespace std;
using namespace arma;

namespace ReLe
{


NESIterationStats::NESIterationStats(unsigned int nbIndividual,
                                     unsigned int nbParams, unsigned int nbEvals)
    : PGPEIterationStats(nbIndividual, nbParams, nbEvals),
      fisherMtx(nbParams, nbParams)
{
}

void NESIterationStats::writeData(ostream &out)
{
    PGPEIterationStats::writeData(out);
    CSVutils::matrixToCSV(fisherMtx, out);
}

}
