#include "parametric/differentiable/NormalPolicy.h"
#include "RandomGenerator.h"

using namespace std;

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// NORMAL POLICY
///////////////////////////////////////////////////////////////////////////////////////

void NormalPolicy::calculateMeanAndStddev(const DenseState& state)
{
    // compute mean value
    arma::vec mean = (*approximator)(state);
    mMean = mean[0];
}

double NormalPolicy::operator()(const DenseState& state, const DenseAction& action)
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

arma::vec NormalPolicy::diff(const DenseState& state, const DenseAction& action)
{
    return (*this)(state,action) * difflog(state,action);
}

arma::vec NormalPolicy::difflog(const DenseState& state, const DenseAction& action)
{
    // get scalar action
    double a = action[0];

    // compute mean value
    calculateMeanAndStddev(state);

    // get features
    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);

    // compute gradient
    return features.t() * (a - mMean) / (mInitialStddev * mInitialStddev);
}

arma::mat NormalPolicy::diff2log(const DenseState& state, const DenseAction& action)
{

    // compute mean value
    calculateMeanAndStddev(state);

    // get features
    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);

    return -features.t() * features / (mInitialStddev * mInitialStddev);

}


///////////////////////////////////////////////////////////////////////////////////////
/// NORMAL POLICY WITH STATE DEPENDANT STDDEV (STD is not a parameter to be learned)
///////////////////////////////////////////////////////////////////////////////////////
void
NormalStateDependantStddevPolicy::calculateMeanAndStddev(const DenseState& state)
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

double MVNPolicy::operator()(const DenseState& state, const DenseAction& action)
{
    UpdateInternalState(state);
    return mvnpdfFast(action, mMean, mCinv, mDeterminant);
}

DenseAction MVNPolicy::operator() (const DenseState& state)
{
    UpdateInternalState(state);
    //TODO controllare assegnamento
    arma::vec v = mvnrandFast(mMean, mCholeskyDec);
    todoAction.copy_vec(v);
    return todoAction;
}

arma::vec MVNPolicy::diff(const DenseState &state, const DenseAction &action)
{
    return (*this)(state,action) * difflog(state,action);
}

arma::vec MVNPolicy::difflog(const DenseState &state, const DenseAction &action)
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
    return 0.5 * features.t() * (mCinv + mCinv.t()) * smdiff;
}

arma::mat MVNPolicy::diff2log(const DenseState &state, const DenseAction &action)
{
    UpdateInternalState(state);

    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);

    // compute gradient
    return - 0.5 * features.t() * (mCinv + mCinv.t()) * features;
}


} //end namespace
