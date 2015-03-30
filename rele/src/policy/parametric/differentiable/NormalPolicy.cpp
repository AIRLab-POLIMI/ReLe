#include "parametric/differentiable/NormalPolicy.h"
#include "RandomGenerator.h"

using namespace std;

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// NORMAL POLICY
///////////////////////////////////////////////////////////////////////////////////////

void NormalPolicy::calculateMeanAndStddev(const arma::vec& state)
{
    // compute mean value
    arma::vec mean = (*approximator)(state);
    mMean = mean[0];
}

double NormalPolicy::operator()(const arma::vec& state, const arma::vec& action)
{

    // compute mean value
    calculateMeanAndStddev(state);

    double scalara = action[0];

    // compute probability
    return ReLe::normpdf(scalara, mMean, mInitialStddev*mInitialStddev);
}

arma::vec NormalPolicy::operator() (const arma::vec& state)
{
    // compute mean value
    calculateMeanAndStddev(state);
    //        MY_PRINT(mMean);
    //        MY_PRINT(mInitialStddev);

    double normn = RandomGenerator::sampleNormal();
    //        MY_PRINT(normn);

    arma::vec output(1);
    output[0] = normn * mInitialStddev + mMean;
    return output;
}

arma::vec NormalPolicy::diff(const arma::vec& state, const arma::vec& action)
{
    return (*this)(state,action) * difflog(state,action);
}

arma::vec NormalPolicy::difflog(const arma::vec& state, const arma::vec& action)
{
    // get scalar action
    double a = action[0];

    // compute mean value
    calculateMeanAndStddev(state);

    // get features
    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);

    // compute gradient
    return features * (a - mMean) / (mInitialStddev * mInitialStddev);
}

arma::mat NormalPolicy::diff2log(const arma::vec& state, const arma::vec& action)
{

    // compute mean value
    calculateMeanAndStddev(state);

    // get features
    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);

    return -features * features.t() / (mInitialStddev * mInitialStddev);

}


///////////////////////////////////////////////////////////////////////////////////////
/// NORMAL POLICY WITH STATE DEPENDANT STDDEV (STD is not a parameter to be learned)
///////////////////////////////////////////////////////////////////////////////////////
void
NormalStateDependantStddevPolicy::calculateMeanAndStddev(const arma::vec& state)
{
    // compute mean value
    arma::vec mean = (*approximator)(state);
    mMean = mean[0];

    // compute stddev
    arma::vec evalstd = (*stdApproximator)(state);
    mInitialStddev = evalstd[0];
}


///////////////////////////////////////////////////////////////////////////////////////
/// MVN POLICY
///////////////////////////////////////////////////////////////////////////////////////

double MVNPolicy::operator()(const arma::vec& state, const arma::vec& action)
{
    UpdateInternalState(state);
    return mvnpdfFast(action, mMean, mCinv, mDeterminant);
}

arma::vec MVNPolicy::operator() (const arma::vec& state)
{
    UpdateInternalState(state);
    return mvnrandFast(mMean, mCholeskyDec);
}

arma::vec MVNPolicy::diff(const arma::vec &state, const arma::vec &action)
{
    return (*this)(state,action) * difflog(state,action);
}

arma::vec MVNPolicy::difflog(const arma::vec &state, const arma::vec &action)
{
    UpdateInternalState(state);

    arma::vec smdiff(mCovariance.n_rows);
    //        std::cout << "Action: ";
    for (unsigned i = 0; i < mCovariance.n_rows; ++i)
    {
        smdiff(i) = action[i] - mMean(i);
        //            std::cout << action[i] << " ";
    }
    //        std::cout << "\n";

    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);

    //        MY_PRINT(features);
    //        MY_PRINT(smdiff);
    //        MY_PRINT(mCinv);

    // compute gradient
    return 0.5 * features * (mCinv + mCinv.t()) * smdiff;
}

arma::mat MVNPolicy::diff2log(const arma::vec &state, const arma::vec &action)
{
    UpdateInternalState(state);

    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);

    // compute gradient
    return - 0.5 * features * (mCinv + mCinv.t()) * features.t();
}


} //end namespace
