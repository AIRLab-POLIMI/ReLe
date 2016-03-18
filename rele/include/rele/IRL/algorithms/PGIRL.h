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

#ifndef PGIRL_H_
#define PGIRL_H_

#include "rele/IRL/IRLAlgorithm.h"
#include "rele/policy/Policy.h"
#include "rele/core/Transition.h"
#include "rele/utils/ArmadilloExtensions.h"

#include <nlopt.hpp>
#include <cassert>

#include "rele/IRL/utils/StepBasedGradientCalculatorFactory.h"

namespace ReLe
{

template<class ActionC, class StateC>
class PlaneGIRL : public IRLAlgorithm<ActionC, StateC>
{
public:

    PlaneGIRL(Dataset<ActionC,StateC>& dataset,
              DifferentiablePolicy<ActionC,StateC>& policy,
              LinearApproximator& rewardFunction,
              double gamma, IrlGrad type)
        : policy(policy), data(dataset), rewardFunction(rewardFunction),
          gamma(gamma)
    {
        nbFunEvals = 0;

        calculator = StepBasedGradientCalculatorFactory<ActionC, StateC>::build(type, rewardFunction.getFeatures(), dataset, policy, gamma);
    }

    virtual ~PlaneGIRL()
    {
        delete calculator;
    }

    void setData(Dataset<ActionC,StateC>& dataset)
    {
        data = dataset;
    }




    virtual void run() override
    {
        int dp = policy.getParametersSize();
        int dr = rewardFunction.getParametersSize();

        arma::mat A = calculator->getGradientDiff();

        A.save("/tmp/ReLe/grad.log", arma::raw_ascii);

        std::cout << "Grads: \n" << A << std::endl;

        ////////////////////////////////////////////////
        /// PRE-PROCESSING
        ////////////////////////////////////////////////
        arma::mat Ared;         //reduced gradient matrix
        arma::uvec nonZeroIdx;  //nonzero elements of the reward weights
        int rnkG = rank(A);
        if ( rnkG < dr && A.n_rows >= A.n_cols )
        {
            // select linearly independent columns
            arma::mat Asub;
            nonZeroIdx = rref(A, Asub);
            std::cout << "Asub: \n" << Asub << std::endl;
            std::cout << "idx: \n" << nonZeroIdx.t()  << std::endl;
            Ared = A.cols(nonZeroIdx);
            assert(rank(Ared) == Ared.n_cols);
        }
        else
        {
            Ared = A;
            nonZeroIdx.set_size(A.n_cols);
            std::iota (std::begin(nonZeroIdx), std::end(nonZeroIdx), 0);
        }

        if(nonZeroIdx.n_elem == 1)
        {
            weights.zeros(A.n_cols);
            weights(nonZeroIdx).ones();
            return;
        }


        Ared.save("/tmp/ReLe/gradRed.log", arma::raw_ascii);

        ////////////////////////////////////////////////
        /// GRAM MATRIX AND NORMAL
        ////////////////////////////////////////////////
        arma::mat gramMatrix = Ared.t() * Ared;
        unsigned int lastr = gramMatrix.n_rows;
        arma::mat X = gramMatrix.rows(0, lastr-2) - arma::repmat(gramMatrix.row(lastr-1), lastr-1, 1);
        X.save("/tmp/ReLe/GM.log", arma::raw_ascii);

        // COMPUTE NULL SPACE
        Y = null(X);
        std::cout << "Y: " << Y << std::endl;
        Y.save("/tmp/ReLe/NullS.log", arma::raw_ascii);

        // prepare the output
        // reset weights
        weights.zeros(A.n_cols);


        if (Y.n_cols > 1)
        {
            ////////////////////////////////////////////////
            /// POST-PROCESSING (IF MULTIPLE SOLUTIONS)
            ////////////////////////////////////////////////

            //setup optimization algorithm
            nlopt::opt optimizator;
            int nbOptParams = Y.n_cols;

            optimizator = nlopt::opt(nlopt::algorithm::LN_COBYLA, nbOptParams);
            optimizator.set_min_objective(PlaneGIRL::wrapper, this);

            unsigned int maxFunEvals = 0;
            nbFunEvals = 0;
            if (maxFunEvals == 0)
                maxFunEvals = std::min(50*nbOptParams, 600);


            optimizator.set_xtol_rel(1e-8);
            optimizator.set_ftol_rel(1e-8);
            optimizator.set_ftol_abs(1e-8);
            optimizator.set_maxeval(maxFunEvals);
            optimizator.add_equality_constraint(PlaneGIRL::wrapper_constr, this, 1e-6);



            //optimize function
            arma::vec wStart(dr, arma::fill::ones);
            wStart /= arma::sum(wStart);
            arma::vec xStart = arma::pinv(Y)*wStart;
            std::vector<double> parameters = arma::conv_to<std::vector<double>>::from(xStart);

            double minf;

            if (optimizator.optimize(parameters, minf) < 0)
            {
                printf("nlopt failed!\n");
                abort();
            }
            else
            {
                arma::vec finalP(nbOptParams);
                for(int i = 0; i < nbOptParams; ++i)
                {
                    finalP(i) = parameters[i];
                }

                weights(nonZeroIdx) = Y*finalP;
            }
        }
        else
        {
            weights(nonZeroIdx) = Y;
        }

        //Normalize (L1) weights
        weights /= arma::sum(weights);

        rewardFunction.setParameters(weights);

    }

    ////////////////////////////////////////////////////////////////
    /// FUNCTIONS FOR THE OPTIMIZATION STEP
    ////////////////////////////////////////////////////////////////
    static double wrapper_constr(unsigned int n, const double* x, double* grad,
                                 void* o)
    {
        return reinterpret_cast<PlaneGIRL*>(o)->oneSumConstraint(n, x, grad);
    }

    double oneSumConstraint(unsigned int n, const double *x, double *grad)
    {
        grad = nullptr;
        arma::vec w(x,n);
        arma::vec p = Y*w;
        double val = arma::norm(p,1) - 1.0;

        if(grad != nullptr)
        {
            arma::vec dS(grad, n, false, true);
            arma::vec dS_w(Y.n_rows, arma::fill::zeros);
            dS = Y.t()*dS_w;
            std::cout << "dS = " << dS.t() << std::endl;
        }

        std::cout << "sum = " << val << std::endl;

        return val;
    }

    static double wrapper(unsigned int n, const double* x, double* grad,
                          void* o)
    {
        return reinterpret_cast<PlaneGIRL*>(o)->objFunction(n, x, grad);
    }

    double objFunction(unsigned int n, const double* x, double* grad)
    {

        ++nbFunEvals;

        arma::vec w(x,n);
        arma::vec p = Y*w;

        if (grad != nullptr)
        {
            abort();
        }

        double norm1_2 = 0.0;
        for (unsigned int i = 0, ie = p.n_elem; i < ie; ++i)
        {
            norm1_2 += sqrt(abs(p(i)));
        }
        norm1_2 *= norm1_2;
        //        std::cerr << norm1_2 << std::endl;
        return norm1_2;

    }

    /*static double wrapperHessian(unsigned int n, const double* x, double* grad,
                                 void* o)
    {
        arma::vec dLambda(grad, n, false, true);
        arma::vec parV(const_cast<double*>(x), n, true);
        double value = reinterpret_cast<PlaneGIRL*>(o)->objFunctionHessian(parV, dLambda, grad != nullptr);

        if(grad != nullptr && arma::norm(dLambda) != 0)
            dLambda /= arma::norm(dLambda);


        std::cout << "x = " << parV.t();
        std::cout << "w = " << parV.t()*reinterpret_cast<PlaneGIRL*>(o)->Y.t();
        if(grad != nullptr)
            std::cout << "dx= " << dLambda.t();
        std::cout << "v = " << value << std::endl;
        return value;
    }

    double objFunctionHessian(const arma::vec& x, arma::vec& dLambda, bool computeGradient)
    {
        ++nbFunEvals;

        arma::vec w = Y*x;


        arma::mat H = calculator.computeHessian(w);

        std::cout << "det(H) = " << arma::det(H) << std::endl;

        arma::vec lambda;
        arma::mat v;

        arma::eig_sym(lambda, v, H);

        if(computeGradient)
        {
            arma::cube Hdiff = calculator.getHessianDiff();
            const arma::vec& v0 = v.col(0);

            arma::vec dLambda_dw(Hdiff.n_slices);

            for(unsigned int s = 0; s < Hdiff.n_slices; s++)
                dLambda_dw(s) = arma::as_scalar(v0.t()*Hdiff.slice(s)*v0);

            dLambda = Y.t()*dLambda_dw;
            std::cout << "dLambda = " << dLambda.t() << std::endl;

        }

        std::cout << "H = " << H;
        std::cout << "eigval = " << lambda.t();
        std::cout << "eigvec = " << v;
        return lambda(0);

    }*/


protected:
    Dataset<ActionC,StateC>& data;
    DifferentiablePolicy<ActionC,StateC>& policy;
    LinearApproximator& rewardFunction;
    double gamma;
    arma::vec weights;
    unsigned int nbFunEvals;
    arma::mat Y;

    GradientCalculator<ActionC,StateC>* calculator;
};


} //end namespace


#endif /* PGIRL_H_ */
