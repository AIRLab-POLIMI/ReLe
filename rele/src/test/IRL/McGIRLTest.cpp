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

#include "Core.h"
#include "PolicyEvalAgent.h"

#include "parametric/differentiable/GibbsPolicy.h"
#include "features/DenseFeatures.h"
#include "features/SparseFeatures.h"
#include "DifferentiableNormals.h"
#include "q_policy/e_Greedy.h"
#include "basis/IdentityBasis.h"
#include "basis/GaussianRbf.h"
#include "basis/PolynomialFunction.h"
#include "basis/ConditionBasedFunction.h"

#include "ParametricRewardMDP.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include "MountainCar.h"
#include "MLE.h"
#include "batch/LSPI.h"
#include "algorithms/PGIRL.h"
#include "nonparametric/RandomPolicy.h"
#include "ArmadilloExtensions.h"

#include <boost/timer/timer.hpp>

using namespace boost::timer;
using namespace std;
using namespace ReLe;
using namespace arma;

class mountain_car_manual_policy : public Policy<FiniteAction, DenseState>
{
public:
    unsigned int operator()(const arma::vec& state)
    {
        if (RandomGenerator::sampleEvent(0.4))
        {
            return RandomGenerator::sampleUniformInt(0,1);
        }
        else
        {
            double speed = state(MountainCar::StateLabel::velocity);
            if (speed <= 0)
                return 0;
            else
                return 2;
        }
    }

    double operator()(const arma::vec& state, const unsigned int& action)
    {
        double speed = state(MountainCar::StateLabel::velocity);
        if (speed <= 0 && action == 0)
            return 1;
        else if(speed > 0 && action == 2)
            return 1;
        else
            return 0;
    }

    // Policy interface
public:
    string getPolicyName()
    {
        return "mountain_car_manual_policy";
    }
    string getPolicyHyperparameters()
    {
        return "";
    }
    string printPolicy()
    {
        return "";
    }

    mountain_car_manual_policy* clone()
    {
        return new mountain_car_manual_policy();
    }
};

class mc_basis : public BasisFunction
{
public:
    mc_basis(arma::vec mu, double sigma_position, double sigma_velocity)
        : mu(mu), sigma_position(sigma_position), sigma_velocity(sigma_velocity)
    {
    }

    virtual void readFromStream(std::istream& in) {}
    virtual void writeOnStream(std::ostream& out) {}

    virtual double operator()(const arma::vec& s)
    {
        int posIdx = MountainCar::StateLabel::position;
        int velIdx = MountainCar::StateLabel::velocity;
        double A = - (s[posIdx] - mu[posIdx]) * (s[posIdx] - mu[posIdx]) / sigma_position;
        double B = - (s[velIdx] - mu[velIdx]) * (s[velIdx] - mu[velIdx]) / sigma_velocity;
        double val = exp(A + B);
        return val;
    }
protected:
    arma::vec mu;
    double sigma_position, sigma_velocity;
};

class mc_reward_bf : public IRLParametricReward<FiniteAction, DenseState>
{
public:
    mc_reward_bf(BasisFunction* basis)
        : phi(*basis)
    {
    }

    double operator()(DenseState& s, FiniteAction& a, DenseState& ns)
    {
        //        if (a.getActionN() != actionIdx)
        //            return 0;

        double val = phi(s);
        return val;
    }

    arma::mat diff(DenseState& s, FiniteAction& a, DenseState& ns)
    {
        return arma::mat();
    }

    static std::vector<IRLParametricReward<FiniteAction, DenseState>*> generate(BasisFunctions& basis)
    {
        std::vector<IRLParametricReward<FiniteAction, DenseState>*> rewards;
        for (int i = 0, ie = basis.size(); i < ie; ++i)
        {
            rewards.push_back(new mc_reward_bf(basis[i]));
        }
        return rewards;
    }

protected:
    BasisFunction& phi;
};

//class mc_reward : public IRLParametricReward<FiniteAction, DenseState>
//{
//public:
//    mc_reward(arma::vec mu, double sigma_position, double sigma_velocity, unsigned int actionIdx)
//        : basis(mu, sigma_position, sigma_velocity), actionIdx(actionIdx)
//    {
//    }


//    double operator()(DenseState& s, FiniteAction& a, DenseState& ns)
//    {
//        //        if (a.getActionN() != actionIdx)
//        //            return 0;

//        double val = basis(s);
//        return val;
//    }

//    arma::mat diff(DenseState& s, FiniteAction& a, DenseState& ns)
//    {
//        return arma::mat();
//    }

//protected:
//    mc_basis basis;
//    unsigned int actionIdx;
//};

//class mc_reward_one : public IRLParametricReward<FiniteAction, DenseState>
//{
//public:
//    mc_reward_one(unsigned int actionIdx)
//        : actionIdx(actionIdx)
//    {
//    }


//    double operator()(DenseState& s, FiniteAction& a, DenseState& ns)
//    {
//        //        if (a.getActionN() != actionIdx)
//        //            return 0;
//        return 1.0;
//    }

//    arma::mat diff(DenseState& s, FiniteAction& a, DenseState& ns)
//    {
//        return arma::mat();
//    }

//protected:
//    unsigned int actionIdx;
//};

int main(int argc, char *argv[])
{
    RandomGenerator::seed(4265436624);

    FileManager fm("mc", "GIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    int nbEpisodes = 100;  // number of episodes simulated to generate mle data
    int budget = 300;      // number of samples used to estimate policy and learn weights

    // define domain
    MountainCar mdp(MountainCar::ConfigurationsLabel::Klein);

    /*** define expert's policy ***/
    mountain_car_manual_policy expertPolicy;

    /*** get expert's trajectories ***/
    PolicyEvalAgent<FiniteAction, DenseState> expert(expertPolicy);
    Core<FiniteAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<FiniteAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();

    /*** save data ***/
    Dataset<FiniteAction,DenseState>& data = collection.data;
    ofstream datafile(fm.addPath("data.log"), ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    data.writeToStream(datafile);
    datafile.close();
    cout << "ended writing expert's data..." << endl;

    /*** define policy for MLE ***/
    vector<FiniteAction> actions;
    for (int i = 0; i < mdp.getSettings().finiteActionDim; ++i)
        actions.push_back(FiniteAction(i));
#if 0
    BasisFunctions basis = PolynomialFunction::generate(1,mdp.getSettings().continuosStateDim);
    //Replicate each basis for the actions (action-1 because the last is linearly dependent on the others)
    BasisFunctions bfs = AndConditionBasisFunction::generate(basis, 2, actions.size()-1);
#else
    BasisFunctions basis = IdentityBasis::generate(1);
    BasisFunctions bfs = AndConditionBasisFunction::generate(basis, 2, actions.size()-1);
#endif

    //create basis vector
    DenseFeatures phi(bfs);
    //create policy
    ParametricGibbsPolicy<DenseState> mlePolicy(actions, phi, 1);

    /*** get only trailing info ***/
    Dataset<FiniteAction,DenseState> dataExpert;
#if 0
    for (int ep = 0; ep < data.size() && budget > 0; ++ep)
    {
        Episode<FiniteAction,DenseState> episodeExpert;
        int nbSamples = data[ep].size();
        for (int t = 0; t < nbSamples && budget > 0; ++t)
        {
            episodeExpert.push_back(data[ep][t]);
            --budget;
        }
        dataExpert.push_back(episodeExpert);
    }
#else
    dataExpert = data;
#endif

    /*** save MLE data ***/
    datafile.open(fm.addPath("mletraining.log"), ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    dataExpert.writeToStream(datafile);
    datafile.close();


    /*** compute MLE ***/
    //    RidgeRegularizedMLE<FiniteAction,DenseState> mle(mlePolicy, dataExpert, 0.1);
    MLE<FiniteAction,DenseState> mle(mlePolicy, dataExpert);
    arma::vec startVal(mlePolicy.getParametersSize(),arma::fill::ones);
    arma::vec pp = mle.solve(startVal);

    std::cerr << pp.t();
    mlePolicy.setParameters(pp);

//    mlePolicy.setTemperature(1.0/0.9);

#if 1//EVAL_MLE
    /*** get mlePolicy trajectories ***/
    PolicyEvalAgent<FiniteAction, DenseState> mleEval(mlePolicy);
    Core<FiniteAction, DenseState> mleCore(mdp, mleEval);
    CollectorStrategy<FiniteAction, DenseState> collectionMle;
    mleCore.getSettings().loggerStrategy = &collectionMle;
    mleCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    mleCore.getSettings().testEpisodeN = nbEpisodes;
    mleCore.runTestEpisodes();


    /*** save data ***/
    Dataset<FiniteAction,DenseState>& dataMle = collectionMle.data;
    datafile.open(fm.addPath("mledata.log"), ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    dataMle.writeToStream(datafile);
    datafile.close();
#endif

    /*** recover reward by IRL (PGIRL) ***/
    std::vector<IRLParametricReward<FiniteAction,DenseState>*> rewards;
//    vec pos_linspace = linspace<vec>(-1.8,1,3);
//    vec vel_linspace = linspace<vec>(-0.1,0.1,3);

    vec pos_linspace = {0.4, 0.0};
    vec vel_linspace = {-0.02,0.02};
    arma::mat yy_vel, xx_pos;
    meshgrid(vel_linspace, pos_linspace, yy_vel, xx_pos);

    arma::vec pos_mesh = vectorise(xx_pos);
    arma::vec vel_mesh = vectorise(yy_vel);
    arma::mat XX = arma::join_horiz(vel_mesh,pos_mesh);


    double sigma_position = 2*pow((0.6+1.2)/4.0,2);
    double sigma_speed    = 2*pow((0.07+0.07)/4.0,2);
    arma::vec widths = {sigma_speed, sigma_position};
    arma::mat WW = repmat(widths, 1, XX.n_rows);
    arma::mat XT = XX.t();

    BasisFunctions rewardBasis = GaussianRbf::generate(XT, WW);
    rewards = mc_reward_bf::generate(rewardBasis);

    assert(rewardBasis.size() == pos_mesh.n_rows);

    int na = 0;
    ofstream bfile(fm.addPath("basis.txt"), ios_base::out);
    //    for(na = 0, nae = actions.size(); na < nae; ++na)
    //    {
    for (int i = 0, ie = pos_mesh.n_rows; i < ie; ++i)
    {
        GaussianRbf* rbf = static_cast<GaussianRbf*>(rewardBasis[i]);
        vec mu = rbf->getCenter();
        vec b = rbf->getWidth();

        //        arma::vec mu(2);
        //        mu(MountainCar::StateLabel::position) = pos_mesh(i);
        //        mu(MountainCar::StateLabel::velocity) = vel_mesh(i);
        //        rewards.push_back(new mc_reward(mu,sigma_position, sigma_speed, na));
        bfile << mu(0) << "\t" << mu(1) << "\t" << b(0) << "\t" << b(1) << std::endl;
    }
    //        rewards.push_back(new mc_reward_one(na));
    //    }
    bfile.close();

    IRLGradType atype = IRLGradType::ENAC;
    PlaneGIRL<FiniteAction,DenseState> pgirl(dataExpert, mlePolicy, rewards,
            mdp.getSettings().gamma, atype);
    cpu_timer timer2;
    timer2.start();
    pgirl.run();
    timer2.stop();
    //    timefile << timer2.format(10, "%w") << std::endl;

    vec rewWeights = pgirl.getWeights();
    rewWeights.save(fm.addPath("rewards.log"),arma::raw_ascii);
    cout << "Weights (plane): " << rewWeights.t();
//    rewWeights = abs(rewWeights);
    //    rewWeights.elem( arma::find(rewWeights < 0) ).zeros();
    //    rewWeights = rewWeights / norm(rewWeights,1);



    vec pos_linspace2 = linspace<vec>(-2.5,1.5,30);
    vec vel_linspace2 = linspace<vec>(-0.2,0.2,30);

    //-- meshgrid
    arma::mat xx_pos2, yy_vel2;
    meshgrid(vel_linspace2, pos_linspace2, yy_vel2, xx_pos2);
    arma::vec pos_mesh2 = vectorise(xx_pos2);
    arma::vec vel_mesh2 = vectorise(yy_vel2);
    //--

    DenseState s(2), sn(2);
    FiniteAction a;

    //    for(na = 0, nae = actions.size(); na < nae; ++na)
    //    {
    a.setActionN(na);
    char gg[100];
    sprintf(gg, "learnedRew_action%d.log", na);
    ofstream rewfilename(fm.addPath(gg), ios_base::out);
    for (int i = 0, ie = pos_mesh2.n_rows; i < ie; ++i)
    {
        s(MountainCar::StateLabel::position) = pos_mesh2(i);
        s(MountainCar::StateLabel::velocity) = vel_mesh2(i);
        vec obj(rewards.size());
        for (int r = 0; r < rewards.size(); ++r)
        {
            IRLParametricReward<FiniteAction,DenseState>& rw = *(rewards[r]);
            obj(r) = rw(s,a,sn);
        }

        double val = dot(obj,rewWeights);
        rewfilename << vel_mesh2(i) << "\t" << pos_mesh2(i) << "\t" << val << endl;
    }
    rewfilename.close();
    //    }


    return 1;

    ofstream dasdsa(fm.addPath("poll.log"), ios_base::out);
    for (int i = 0, ie = pos_mesh2.n_rows; i < ie; ++i)
    {
        s(MountainCar::StateLabel::position) = pos_mesh2(i);
        s(MountainCar::StateLabel::velocity) = vel_mesh2(i);
        FiniteAction dd;
        dd.setActionN(1);
        double val = mlePolicy(s,dd);
        dasdsa << vel_mesh2(i) << "\t" << pos_mesh2(i) << "\t" << val << endl;
    }
    dasdsa.close();



    /*** LSPI ***/
    //define basis for Q-function
    //    BasisFunctions qbasis = GaussianRbf::generate({7,7},{-0.07, 0.07, -1.2, 0.5});
    BasisFunctions qbasis = GaussianRbf::generate(XT, WW);
    BasisFunctions qbasisrep = AndConditionBasisFunction::generate(qbasis, 2, actions.size());
    //create basis vector
    DenseFeatures qphi(qbasisrep);
    //    cout << qphi.rows() << " " << qphi.cols() << endl;
    //    for (int i = 0, ie = pos_mesh.n_rows; i < ie; ++i)
    //    {
    //        arma::vec mu(2);
    //        mu(MountainCar::StateLabel::position) = pos_mesh(i);
    //        mu(MountainCar::StateLabel::velocity) = vel_mesh(i);
    //        qbasis.push_back(new mc_basis(mu, sigma_position, sigma_speed));
    //    }
    //    qbasis.push_back(new PolynomialFunction());
    //    BasisFunctions qbasisrep;
    //    for (int i = 0, ie = actions.size(); i < ie; ++i)
    //    {
    //        for (int k = 0, ke = qbasis.size(); k < ke; ++k)
    //        {
    //            qbasisrep.push_back(new AndConditionBasisFunction(qbasis[k],2,i));
    //        }
    //    }
    //    //create basis vector
    //    DenseFeatures qphi(qbasisrep);
    //create random policy
    StochasticDiscretePolicy<FiniteAction,DenseState> randomPolicy(actions);

    //set random start
    mdp.s0type = MountainCar::ConfigurationsLabel::Random;

    //create data per lspi
    PolicyEvalAgent<FiniteAction, DenseState> lspiEval(randomPolicy);
    Core<FiniteAction, DenseState> lspiCore(mdp, lspiEval);
    CollectorStrategy<FiniteAction, DenseState> collectionlspi;
    lspiCore.getSettings().loggerStrategy = &collectionlspi;
    lspiCore.getSettings().episodeLenght = 10;
    lspiCore.getSettings().testEpisodeN = 1000;
    lspiCore.runTestEpisodes();


    /*** save data ***/
    Dataset<FiniteAction,DenseState>& dataLSPI = collectionlspi.data;
    //update with computed reward
    for (int i = 0, ie = dataLSPI.size(); i < ie; ++i)
    {
        vec obj(rewards.size());
        int nbs = dataLSPI[i].size();
        for (int t = 0; t < nbs; ++t)
        {
            Transition<FiniteAction, DenseState>& tr = dataLSPI[i][t];
            for (int r = 0; r < rewards.size(); ++r)
            {
                IRLParametricReward<FiniteAction,DenseState>& rw = *(rewards[r]);
                obj(r) = rw(tr.x, tr.u, tr.xn);
            }

            double val = dot(obj,rewWeights);
            tr.r[0] = val;
        }
    }

    datafile.open(fm.addPath("lspidata.log"), ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    dataLSPI.writeToStream(datafile);
    datafile.close();

    e_GreedyApproximate lspiPolicy;
    lspiPolicy.setEpsilon(0.01);
    lspiPolicy.setNactions(actions.size());
    LSPI<FiniteAction> lspi(dataLSPI, lspiPolicy, qphi, mdp.getSettings().gamma);
    lspi.run(40,1e-8);

    cout << dynamic_cast<LinearApproximator*>(lspiPolicy.getQ())->getParameters() << endl;

    LinearApproximator& la = *(dynamic_cast<LinearApproximator*>(lspiPolicy.getQ()));
    for (na = 0; na < 3; ++na)
    {
        char ff[100];
        sprintf(ff, "learnedQ_a%d.log",na);
        ofstream asdsa(fm.addPath(ff), ios_base::out);
        for (int i = 0, ie = pos_mesh2.n_rows; i < ie; ++i)
        {
            vec inp = {vel_mesh2(i), pos_mesh2(i), na};
            arma::mat val = la(inp);
            asdsa << vel_mesh2(i) << "\t" << pos_mesh2(i) << "\t" << val(0,0) <<std::endl;
        }
        asdsa.close();
    }

    lspiPolicy.setEpsilon(0.0);

    //create data per lspi
    PolicyEvalAgent<FiniteAction, DenseState> finalEval(lspiPolicy);
    Core<FiniteAction, DenseState> finalCore(mdp, finalEval);
    CollectorStrategy<FiniteAction, DenseState> collectionFinal;
    finalCore.getSettings().loggerStrategy = &collectionFinal;
    finalCore.getSettings().episodeLenght = 150;
    finalCore.getSettings().testEpisodeN = 1000;
    finalCore.runTestEpisodes();


    /*** save data ***/
    Dataset<FiniteAction,DenseState>& dataFinal = collectionFinal.data;
    datafile.open(fm.addPath("finaldata.log"), ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    dataFinal.writeToStream(datafile);
    datafile.close();

    ofstream policyFile(fm.addPath("finalpol.log"), ios_base::out);
    for (int i = 0, ie = pos_mesh2.n_rows; i < ie; ++i)
    {
        s(MountainCar::StateLabel::position) = pos_mesh2(i);
        s(MountainCar::StateLabel::velocity) = vel_mesh2(i);
        unsigned int act = lspiPolicy(s);
        policyFile << vel_mesh2(i) << "\t" << pos_mesh2(i) << "\t" << act << endl;
    }
    policyFile.close();



    return 0;
}
