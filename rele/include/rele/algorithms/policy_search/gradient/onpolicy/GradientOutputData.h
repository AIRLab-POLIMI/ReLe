#ifndef GRADIENTOUTPUTDATA_H_
#define GRADIENTOUTPUTDATA_H_

#include "Basics.h"

namespace ReLe
{

class GradientIndividual : virtual public AgentOutputData
{
public:

    GradientIndividual();

    virtual ~GradientIndividual()
    {}

    virtual void writeData(std::ostream& os);

    virtual void writeDecoratedData(std::ostream& os);

    std::vector<double> history_J;
    std::vector<arma::vec> history_gradients;
    arma::vec policy_parameters;
    arma::vec estimated_gradient;
    arma::vec stepLength;
};


}//end namespace

#endif // GRADIENTOUTPUTDATA_H_
