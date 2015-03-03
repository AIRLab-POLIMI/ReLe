#include "parametric/differentiable/NormalPolicy.h"
#include "RandomGenerator.h"

using namespace std;

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// NORMAL POLICY
///////////////////////////////////////////////////////////////////////////////////////

void NormalPolicy::calculateMeanAndStddev(const DenseState &state)
{
    // compute mean value
    arma::vec mean = (*approximator)(state);
    mMean = mean[0];
}

double NormalPolicy::operator()(const DenseState &state, const DenseAction &action)
{

    // compute mean value
    calculateMeanAndStddev(state);

    double scalara = action[0];

    // compute probability
    return ReLe::normpdf(scalara, mMean, mInitialStddev*mInitialStddev);
}

DenseAction NormalPolicy::operator() (const DenseState& state)
{
    // compute mean value
    calculateMeanAndStddev(state);
    //        MY_PRINT(mMean);
    //        MY_PRINT(mInitialStddev);

    double normn = RandomGenerator::sampleNormal();
    //        MY_PRINT(normn);

    todoAction[0] = normn * mInitialStddev + mMean;
    return todoAction;
}

arma::vec NormalPolicy::diff(const DenseState &state, const DenseAction &action)
{
    return (*this)(state,action) * difflog(state,action);
}

arma::vec NormalPolicy::difflog(const DenseState &state, const DenseAction &action)
{
    // get scalar action
    double a = action[0];

    // compute mean value
    calculateMeanAndStddev(state);

    // get features
    AbstractBasisVector& basis = approximator->getBasis();
    arma::vec features = basis(state);

    // compute gradient
    return features.t() * (a - mMean) / (mInitialStddev * mInitialStddev);
}

arma::mat NormalPolicy::diff2log(const DenseState &state, const DenseAction &action)
{

    // compute mean value
    calculateMeanAndStddev(state);

    // get features
    AbstractBasisVector& basis = approximator->getBasis();
    arma::vec features = basis(state);

    return -features.t() * features / (mInitialStddev * mInitialStddev);

}

} //end namespace
