#include "policy_search/NES/NESOutputData.h"

using namespace std;
using namespace arma;

namespace ReLe
{


xNESIterationStats::xNESIterationStats(unsigned int nbIndividual,
                                       unsigned int nbParams, unsigned int nbEvals)
    : PGPEIterationStats(nbIndividual, nbParams, nbEvals),
      fisherMtx(nbParams, nbParams)
{
}

void xNESIterationStats::writeData(ostream &out)
{
    PGPEIterationStats::writeData(out);
    int i, ie, j, je;
    for (i = 0, ie = fisherMtx.n_rows; i < ie; ++i)
        for (j = 0, je = fisherMtx.n_cols; j < je; ++j)
            out << fisherMtx(i,j) << "\t";
}

}
