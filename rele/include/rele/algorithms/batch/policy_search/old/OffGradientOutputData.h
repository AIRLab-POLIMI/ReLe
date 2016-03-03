#ifndef OFFGRADIENTOUTPUTDATA_H_
#define OFFGRADIENTOUTPUTDATA_H_

#include "rele/core/Basics.h"
#include "rele/algorithms/policy_search/gradient/GradientOutputData.h"

namespace ReLe
{

class OffGradientIndividual : public GradientIndividual
{
public:

    OffGradientIndividual();

    virtual ~OffGradientIndividual()
    {}

    virtual void writeData(std::ostream& os) override;

    virtual void writeDecoratedData(std::ostream& os) override;

    std::vector<double> history_impWeights;
};


}//end namespace

#endif // OFFGRADIENTOUTPUTDATA_H_
