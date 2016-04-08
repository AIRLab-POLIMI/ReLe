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
#include "rele/core/BatchCore.h"
#include "rele/core/PolicyEvalAgent.h"

#include "rele/policy/parametric/differentiable/GenericGibbsPolicy.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/features/SparseFeatures.h"
#include "rele/statistics/DifferentiableNormals.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/basis/ConditionBasedFunction.h"

#include "rele/IRL/ParametricRewardMDP.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/FileManager.h"

#include "rele/environments/MountainCar.h"
#include "rele/algorithms/batch/td/LSPI.h"
#include "rele/utils/ArmadilloExtensions.h"
#include "rele/policy/nonparametric/RandomPolicy.h"

#include <boost/timer/timer.hpp>

using namespace boost::timer;
using namespace std;
using namespace ReLe;
using namespace arma;

class mc_basis : public BasisFunction
{
public:
    mc_basis(arma::vec mu, double sigma_position, double sigma_velocity)
        : mu(mu), sigma_position(sigma_position), sigma_velocity(sigma_velocity)
    {
    }

    virtual void readFromStream(std::istream& in) override {}
    virtual void writeOnStream(std::ostream& out) override {}

    virtual double operator()(const arma::vec& s) override
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

/*class mc_reward_bf : public IRLParametricReward<FiniteAction, DenseState>
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
};*/

int main(int argc, char *argv[])
{
    vector<FiniteAction> actions;
    for (int i = 0; i < 3; ++i)
        actions.push_back(FiniteAction(i));

    /*** define basis ***/
    vec pos_linspace = linspace<vec>(-1.2,0.6,7);
    vec vel_linspace = linspace<vec>(-0.07,0.07,7);

    //-- meshgrid
    arma::mat yy_vel, xx_pos;
    meshgrid(vel_linspace, pos_linspace, yy_vel, xx_pos);
    //--

    arma::vec pos_mesh = vectorise(xx_pos);
    arma::vec vel_mesh = vectorise(yy_vel);
    arma::mat XX = arma::join_horiz(vel_mesh,pos_mesh);

    double sigma_position = 2*pow((0.6+1.2)/10.,2);
    double sigma_speed    = 2*pow((0.07+0.07)/10.,2);
    arma::vec widths = {sigma_speed, sigma_position};
    arma::mat WW = repmat(widths, 1, XX.n_rows);
    arma::mat XT = XX.t();

    BasisFunctions qbasis = GaussianRbf::generate(XT, WW);
    qbasis.push_back(new PolynomialFunction());
    BasisFunctions qbasisrep = AndConditionBasisFunction::generate(qbasis, 2, actions.size());
    //create basis vector
    DenseFeatures qphi(qbasisrep);

    // vec x = {-0.03,0.1,0};
    // vec dd = qphi(x);
    // dd.save(fm.addPath("cbasis.dat"), arma::raw_ascii);
    // return 1;

    /*** load data ***/
    ifstream is("mc_lspi_data.dat");
    Dataset<FiniteAction, DenseState> dataLSPI;
    dataLSPI.readFromStream(is);
    is.close();

    ofstream of("data.log");
    of << std::setprecision(OS_PRECISION);
    for (auto ep : dataLSPI)
    {
        for (auto tr : ep)
        {
            int a = tr.u - 1;
            of << tr.x(0) << " " << tr.x(1) << " ";
            of << a << " " << tr.xn(0) << " " << tr.xn(1) << " " << tr.r[0] << endl;
        }
    }
    of.close();

    e_GreedyApproximate lspiPolicy;
    lspiPolicy.setEpsilon(-0.0);
    lspiPolicy.setNactions(actions.size());
    LSPI<FiniteAction> batchAgent(dataLSPI, lspiPolicy, qphi, 0.01);

    auto&& core = buildBatchOnlyCore(dataLSPI, batchAgent);

    core.getSettings().maxBatchIterations = 100;

    double gamma = 0.9;
    EnvironmentSettings envSettings;
    envSettings.gamma = gamma;

    core.run(envSettings);

//    cout << dynamic_cast<LinearApproximator*>(lspiPolicy.getQ())->getParameters() << endl;

    // define domain
    MountainCar mdp(MountainCar::ConfigurationsLabel::Random);

    PolicyEvalAgent<FiniteAction, DenseState> finalEval(lspiPolicy);
    Core<FiniteAction, DenseState> finalCore(mdp, finalEval);
    CollectorStrategy<FiniteAction, DenseState> collectionFinal;
    finalCore.getSettings().loggerStrategy = &collectionFinal;
    finalCore.getSettings().episodeLength = 250;
    finalCore.getSettings().testEpisodeN = 1000;
    finalCore.runTestEpisodes();


    /*** save data ***/
    Dataset<FiniteAction,DenseState>& dataFinal = collectionFinal.data;
    ofstream datafile("finaldata.log", ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    dataFinal.writeToStream(datafile);
    datafile.close();

    return 0;
}
