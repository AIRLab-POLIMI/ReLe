#ifndef OFFGRADIENTOUTPUTDATA_H_
#define OFFGRADIENTOUTPUTDATA_H_

#include "Basics.h"
#include "policy_search/gradient/onpolicy/GradientOutputData.h"

namespace ReLe
{

class OffGradientIndividual : public GradientIndividual
{
public:

    OffGradientIndividual();

    virtual ~OffGradientIndividual()
    {}

    virtual void writeData(std::ostream& os);

    virtual void writeDecoratedData(std::ostream& os);

    std::vector<double> history_impWeights;
};


}//end namespace

#endif // OFFGRADIENTOUTPUTDATA_H_
