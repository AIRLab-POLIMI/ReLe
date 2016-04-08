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

#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/policy/parametric/differentiable/GenericNormalPolicy.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/basis/SubspaceBasis.h"
#include "rele/approximators/basis/ModularBasis.h"
#include "rele/approximators/basis/NormBasis.h"
#include "rele/approximators/features/SparseFeatures.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/FileManager.h"

#include "rele/environments/Pursuer.h"
#include "PGTest.h"

using namespace std;
using namespace ReLe;
using namespace arma;

struct WallNearBasis: public BasisFunction
{
public:
    enum dir {N, S, W, E};

public:
    WallNearBasis(dir wall, double threshold) : wall(wall), threshold(threshold)
    {

    }

    virtual double operator()(const arma::vec& input) override
    {
        switch(wall)
        {
        case N:
            return abs(input[Pursuer::y] - 10) < threshold;

        case S:
            return abs(input[Pursuer::y] + 10) < threshold;

        case W:
            return abs(input[Pursuer::x] + 10) < threshold;

        case E:
            return abs(input[Pursuer::x] + 10) < threshold;

        default:
            return 0;
        }

    }

    virtual void writeOnStream(std::ostream& out) override
    {

    }

    virtual void readFromStream(std::istream& in) override
    {

    }

    ~WallNearBasis()
    {

    }

private:
    dir wall;
    double threshold;
};

class PursuerDirectionBasis: public BasisFunction
{
public:
    virtual double operator()(const arma::vec& input) override
    {
        return RangePi::wrap(atan2(input[Pursuer::yp], input[Pursuer::xp]));
    }

    virtual void writeOnStream(std::ostream& out) override
    {

    }

    virtual void readFromStream(std::istream& in) override
    {

    }

    virtual ~PursuerDirectionBasis()
    {

    }
};

int main(int argc, char *argv[])
{
    CommandLineParser clp;
    gradConfig config = clp.getConfig(argc, argv);
    config.envName = "pursuer";

    Pursuer mdp;

    int dim = mdp.getSettings().stateDimensionality;

    //--- define policy (low level)
    BasisFunctions basis;

    basis.push_back(new SubspaceBasis(new NormBasis(), arma::span(Pursuer::xp, Pursuer::yp)));
    basis.push_back(new SubspaceBasis(new NormBasis(), arma::span(Pursuer::x, Pursuer::y)));
    basis.push_back(new ModularDifference(Pursuer::theta, Pursuer::thetap, RangePi()));
    basis.push_back(new PursuerDirectionBasis());
    double criticalDistance = 0.5;
    basis.push_back(new WallNearBasis(WallNearBasis::N, criticalDistance));
    basis.push_back(new WallNearBasis(WallNearBasis::S, criticalDistance));
    basis.push_back(new WallNearBasis(WallNearBasis::W, criticalDistance));
    basis.push_back(new WallNearBasis(WallNearBasis::E, criticalDistance));



    SparseFeatures phi(basis, 2);

    arma::mat cov(2, 2, arma::fill::eye);
    cov *= 0.1;
    MVNPolicy policy(phi, cov);
    //SaturatedRegressor regressor(phi, {0, -M_PI}, {1, M_PI});
    //GenericMVNPolicy policy(regressor, cov);
    //---

    PGTest<DenseAction, DenseState> pgTest(config, mdp, policy);
    pgTest.run();

    /*delete core.getSettings().loggerStrategy;

    //--- collect some trajectories
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        fm.addPath(outputname),  WriteStrategy<DenseAction, DenseState>::TRANS,false);
    core.getSettings().testEpisodeN = 3000;
    core.runTestEpisodes();
    //---

    cout << "Learned Parameters: " << endl;
    cout << policy.getParameters().t();

    delete core.getSettings().loggerStrategy;*/

    return 0;
}
