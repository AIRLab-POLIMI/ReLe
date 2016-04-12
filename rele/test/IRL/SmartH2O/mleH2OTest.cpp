/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
 * Versione 1.0
 *
 * This file is part of rele.
 *
 * rele is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rele is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with rele.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "rele/core/Core.h"
#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/policy/parametric/differentiable/GenericNormalPolicy.h"
#include "rele/policy/parametric/differentiable/ParametricMixturePolicy.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/statistics/DifferentiableNormals.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/basis/PolynomialFunction.h"

#include "rele/environments/LQR.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/IRL/algorithms/GIRL.h"
#include "rele/algorithms/policy_search/PGPE/PGPE.h"
#include "rele/IRL/ParametricRewardMDP.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/FileManager.h"

#include "rele/algorithms/policy_search/gradient/FunctionGradient.h"
#include "rele/algorithms/policy_search/gradient/PolicyGradientAlgorithm.h"

#include <boost/timer/timer.hpp>

#include "rele/policy/utils/MLE.h"

using namespace boost::timer;
using namespace std;
using namespace ReLe;
using namespace arma;

//#define NORMALDIAG
#define NORMALSTDSTATE_LEARN
//#define GENERICNORMALSTDSTATE


class HourFeat : public DenseFeatures_<arma::vec>
{
public:
    HourFeat() : DenseFeatures_(nullptr)
    {
    }

    virtual ~HourFeat()
    {
    }

    virtual arma::mat operator()(const arma::vec& input) override
    {
        arma::mat output(rows(), cols());
        output.zeros();
        output(input(0), 0) = 1;
        //output(24) = 1;

        return output;
    }

    inline virtual size_t rows() const override
    {
        return 24;
    }

    inline virtual size_t cols() const override
    {
        return 1;
    }

};

void help()
{
    cout << "lqr_GMGIRL [algorithm]" << endl;
    cout << " - algorithm: r, rb, g, gb (default)" << endl;
}

int main(int argc, char *argv[])
{
    //    RandomGenerator::seed(45423424);
    // RandomGenerator::seed(8763575);
    FileManager fm("SmartH2O", "MLE");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);


    Dataset<DenseAction,DenseState> data;
    ifstream inputData(argv[1]);
    if (inputData.is_open())
    {
        data.readFromStream(inputData);
    }
    inputData.close();
    cerr << "#episodes: " << data.size() << endl;

    ofstream outputData(fm.addPath("read_test.csv"));
    data.writeToStream(outputData);
    outputData.close();

    HourFeat phi;

#ifdef NORMALDIAG
    MVNDiagonalPolicy policy(phi);

    double vv[] =
    {
        9.7326,
        151.8182,
        51.9251,
        164.0107,
        171.4439,
        225.0267,
        185.2941,
        168.8235,
        157.8610,
        129.1444,
        120.5348,
        110.5348,
        108.8770,
        135.4545,
        170.1070,
        197.4332,
        181.3904,
        183.7433,
        152.8342,
        102.7807,
        54.0642,
        40.8021,
        20.6417,
        10.8556
    };
    //arma::vec startVal(vv, 24);
    arma::vec startVal(25);
    startVal.ones();
    startVal(24) = 8000;
#else
#ifdef NORMALSTDSTATE_LEARN
    NormalLearnableStateDependantStddevPolicy policy(phi,phi);

    arma::vec startMeanWeights(phi.rows(), arma::fill::ones);
    arma::vec startStdWeights(phi.rows(), arma::fill::ones);
    startStdWeights *= 400;
    arma::vec startVal = vectorize(startMeanWeights, startStdWeights);
#else
#ifdef GENERICNORMALSTDSTATE
    LinearApproximator meanReg(phi);
    LinearApproximator stdReg(phi);
    GenericMVNStateDependantStddevPolicy policy(meanReg, stdReg);

    arma::vec startMeanWeights(phi.rows(), arma::fill::ones);
    arma::vec startStdWeights(phi.rows(), arma::fill::ones);
    startStdWeights *= 400;
    arma::vec startVal = vectorize(startMeanWeights, startStdWeights);
#else
    LinearApproximator meanReg_b1(phi);
    LinearApproximator meanReg_b2(phi);

    LinearApproximator stdReg_b1(phi);
    LinearApproximator stdReg_b2(phi);

    std::vector<DifferentiablePolicy<DenseAction,DenseState>*> mixture;
    mixture.push_back(new GenericMVNStateDependantStddevPolicy(meanReg_b1, stdReg_b1));
    mixture.push_back(new GenericMVNStateDependantStddevPolicy(meanReg_b2, stdReg_b2));

    GenericParametricLogisticMixturePolicy<DenseAction,DenseState> policy(mixture);
//    arma::vec startMeanWeights(phi.rows(), arma::fill::randu);
    arma::vec startStdWeights(phi.rows(), arma::fill::randu);
    startStdWeights *= 200;

    arma::vec startAlphaWeights(1, arma::fill::ones);
    startAlphaWeights /= 2;
    arma::vec startVal = vectorize(arma::randu(phi.rows(),1), startStdWeights);
    startVal = vectorize(startVal, arma::randu(phi.rows(),1), startStdWeights);
    startVal = vectorize(startVal, startAlphaWeights);
#endif
#endif
#endif

    cout << endl << "Policy: " << policy.getPolicyName() << endl << endl;

    MLE<DenseAction,DenseState> mle(policy, data);
    mle.compute(startVal, 400);
    arma::vec pp = policy.getParameters();

    std::cerr << endl << "MLE Params (" << pp.n_elem << " weights): " << endl << pp.t();
    policy.setParameters(pp);

    ofstream outdata(fm.addPath("mlepol.dat"));
    for (auto ep : data)
        for (auto tr : ep)
        {
            auto st = tr.x;
            auto ac = tr.u;

            auto pac = policy(st);

            outdata << pac(0) << "\t" << ac(0) << endl;
        }
    outdata.close();

    //write MATLAB test
    ofstream matlabTout(fm.addPath("checkMLE.m"));
    if (matlabTout.is_open())
    {
        matlabTout << "A = dlmread('" << fm.addPath("mlepol.dat") << "');" << endl;
        matlabTout << "Q1 = reshape (A(1:end-23,1), 24, size(A(1:end-23,1),1)/24);" << endl;
        matlabTout << "Q2 = reshape (A(1:end-23,2), 24, size(A(1:end-23,1),1)/24);" << endl;
        matlabTout << "fprintf('# Per-Hour Action Mean (mle fit, original data)\\n\\n')" << endl;
        matlabTout << "disp([mean(Q1,2), mean(Q2,2)])" << endl;
        matlabTout << "fprintf('# Per-Hour Action Standard Deviation (mle fit, original data)\\n\\n')" << endl;
        matlabTout << "disp([std(Q1,[],2), std(Q2,[],2)])" << endl;
        matlabTout << "exit" << endl;
        matlabTout.close();
        char cmd[900];
        sprintf(cmd, "matlab2015b -nodesktop -nojvm -nosplash -r \"run('%s')\"", fm.addPath("checkMLE.m").c_str());
        system(cmd);
    }

    /**
    IdentityBasis* pf = new IdentityBasis(0);
    DenseFeatures phi(pf);
    NormalPolicy tmpPolicy(1, phi);
    MLE<DenseAction,DenseState> mle(policy, data);
    double vv[] = {0,0,0,0,0};
    arma::vec startVal(vv,5);
    //    double vv[] = {0,0,0,0};
    //    arma::vec startVal(vv,4);
    arma::vec pp = mle.solve(startVal);

    std::cerr << pp.t();
    policy.setParameters(pp);

    int count = 0;
    arma::mat F;
    for (int ep = 0; ep < nbEpisodes; ++ep) //TO AVOID OVERFITTING
    {
        int nbSteps = data[ep].size();
        for (int t = 0; t < nbSteps; ++t)
        {
            Transition<DenseAction, DenseState>& tr = data[ep][t];
            arma::vec aa = policy(tr.x);
            F = arma::join_horiz(F,aa);
            ++count;
        }
    }

    F.save(fm.addPath("datafit.log"), arma::raw_ascii);

    //    //compute importance weights
    //    arma::vec IWOrig;
    //    for (int ep = 0; ep < nbEpisodes; ++ ep)
    //    {
    //        int nbSteps = data[ep].size();
    //        for (int t = 0; t < nbSteps; ++t)
    //        {
    //            Transition<DenseAction, DenseState>& tr = data[ep][t];
    //            arma::vec iw(1);
    //            iw(0) = policy(tr.x,tr.u)/tmpPolicy(tr.x,tr.u);
    //            IWOrig = arma::join_vert(IWOrig, iw);
    //        }
    //    }
    //    IWOrig.save(fm.addPath("iworig.log"), arma::raw_ascii);


    LQR_1D_WS rewardRegressor;
    GIRL<DenseAction,DenseState> irlAlg(data, policy, rewardRegressor,
                                        mdp.getSettings().gamma, atype);

    ofstream timefile(fm.addPath("timer.log"));


    //Run MWAL
    cpu_timer timer;
    timer.start();
    irlAlg.run();
    timer.stop();
    arma::vec gnormw = irlAlg.getWeights();

    timefile << timer.format(10, "%w") << std::endl;

    cout << "Weights (gnorm): " << gnormw.t();

    char name[100];
    sprintf(name, "girl_gnorm_%s.log", gtypestr);
    ofstream outf(fm.addPath(name), std::ofstream::out);
    outf << std::setprecision(OS_PRECISION);
    for (int i = 0; i < gnormw.n_elem; ++i)
    {
        outf << gnormw[i] << " ";
    }
    outf.close();
    **/

    return 0;
}
