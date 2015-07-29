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

#include "features/SparseFeatures.h"
#include "features/DenseFeatures.h"
#include "regressors/LinearApproximator.h"
#include "basis/QuadraticBasis.h"
#include "basis/IdentityBasis.h"

#include "parametric/differentiable/NormalPolicy.h"

#include "LQR.h"
#include "LQRsolver.h"
#include "PolicyEvalAgent.h"
#include "algorithms/GIRL.h"
#include "algorithms/PGIRL.h"

#include "FileManager.h"

using namespace std;
using namespace arma;
using namespace ReLe;

class LQR_RewardBasis : public BasisFunction
{
public:
    LQR_RewardBasis(unsigned int i, unsigned int dim)
    {
        mat Q(dim, dim);
        mat R(dim, dim);

        double e = 0.1;
        for (int j = 0; j < dim; j++)
        {
            if (i == j)
            {
                Q(j,j) = 1.0 - e;
                R(j,j) = e;
            }
            else
            {
                Q(j,j) = e;
                R(j,j) = 1.0 - e;
            }
        }

        bf1 = new QuadraticBasis(Q, span(0, dim-1));
        bf2 = new QuadraticBasis(R, span(dim,2*dim-1));
    }

    virtual double operator()(const vec& input)
    {
        return -(*bf1)(input)-(*bf2)(input);
    }

    virtual void writeOnStream(std::ostream& out)
    {

    }

    virtual void readFromStream(std::istream& in)
    {

    }

private:
    BasisFunction* bf1;
    BasisFunction* bf2;
};

int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);

    IRLGradType atype = IRLGradType::GB;
    vec eReward = {0.2, 0.7, 0.1};
    int nbEpisodes = 2000;

    FileManager fm("lqr", "GIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /* Learn lqr correct policy */
    int dim = eReward.n_elem;
    LQR mdp(dim, dim);

    BasisFunctions basis;
    for (int i = 0; i < dim; ++i)
    {
        basis.push_back(new IdentityBasis(i));
    }

    SparseFeatures phi;
    phi.setDiagonal(basis);

    MVNPolicy expertPolicy(phi);

    /*** solve the problem in exact way ***/
    LQRsolver solver(mdp,phi);
    solver.setRewardWeights(eReward);
    mat K = solver.computeOptSolution();
    arma::vec p = K.diag();
    expertPolicy.setParameters(p);

    std::cout << "Rewards: ";
    for (int i = 0; i < eReward.n_elem; ++i)
    {
        std::cout << eReward(i) << " ";
    }
    std::cout << "| Params: " << expertPolicy.getParameters().t() << std::endl;


    PolicyEvalAgent<DenseAction, DenseState> expert(expertPolicy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    /* Create parametric reward */
    BasisFunctions basisReward;
    for(unsigned int i = 0; i < eReward.n_elem; i++)
        basisReward.push_back(new LQR_RewardBasis(i, dim));
    DenseFeatures phiReward(basisReward);


    LinearApproximator rewardRegressor(phiReward);
    GIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor,
                                        mdp.getSettings().gamma, atype);

    PlaneGIRL<DenseAction, DenseState> irlAlg2(data, expertPolicy, basisReward,
            mdp.getSettings().gamma, atype);


    //Run GIRL
    irlAlg.run();
    arma::vec gnormw = irlAlg.getWeights();

    //Run PGIRL
    irlAlg2.run();
    arma::vec planew = irlAlg2.getWeights();


    //Print results
    cout << "Weights (gnorm): " << gnormw.t();
    cout << "Weights (plane): " << planew.t();

    return 0;
}
