/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_CSI_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_CSI_H_

namespace ReLe
{

template<class StateC>
class CSI : public IRLAlgorithm<FiniteAction, StateC>
{
public:
    CSI(Dataset<FiniteAction, StateC>& data, Dataset<FiniteAction, StateC>& dataR,
        LinearApproximator& rewardf, double gamma, unsigned int nActions, bool heuristic = true) :
        data(data), rewardf(rewardf), gamma(gamma), nActions(nActions), heuristic(heuristic)
    {
        lambda_r = 1e-5;
        epsilon = 1e-3;
        N_final = 20;
        lambda_RCAL = 0;
    }

    virtual void run() override
    {
        arma::mat phiData, phiDataNext, phiAll, phiAllNext;
        computeFeatures(phiData, phiDataNext, phiAll, phiAllNext);

        //TODO different dataset for regression
        arma::mat& phiData_r = phiData;
        arma::mat& phiAllNext_r = phiAllNext;

        //TODO different features for regression
        arma::mat& psiData_r = phiData_r;

        arma::vec theta_c = RCAL(phiData, phiData_r, phiAllNext_r);
        arma::vec Q_r = phiData_r*theta_c;
        arma::mat Q_r_next = phiAllNext_r*theta_c;
        arma::vec Q_r_max = arma::max(Q_r_next, 1);
        arma::vec data_regression = Q_r-gamma*Q_r_max;

        regression(psiData_r, data_regression);

    }

    virtual ~CSI()
    {

    }

private:
    void computeFeatures(arma::mat& phiData, arma::mat& phiDataNext, arma::mat& phiAll, arma::mat& phiAllNext)
    {
        unsigned int nTransitions = data.getTransitionsNumber() - data.size();
        unsigned int nFeatures = rewardf.getParametersSize();

        Features& phi = rewardf.getFeatures();

        phiData.set_size(nTransitions, nFeatures);
        phiDataNext.set_size(nTransitions, nFeatures);
        phiAll.set_size(nTransitions*nActions, nFeatures);

        unsigned int count = 0;
        for(auto& episode : data)
        {
            for(unsigned t = 0; t + 1 < episode.size(); t++)
            {
                auto& tr = episode[t];
                auto& trNext = episode[t+1];
                phiData.row(count) = phi(tr.x, tr.u).t();
                phiDataNext.row(count) = phi(trNext.x, trNext.u).t();

                for (unsigned int u = 0; u < nActions; u++)
                {
                    phiAll.row(count + nTransitions*u) = phi(tr.x, FiniteAction(u)).t();
                    phiAllNext.row(count + nTransitions*u) = phi(trNext.x, FiniteAction(u)).t();
                }

                count++;
            }
        }
    }

    arma::vec RCAL(const arma::mat& phi, const arma::mat& phi_r, const arma::mat& phi_r_next)
    {
        unsigned int nTransitions_c = data.getTransitionsNumber() - data.size();
        unsigned int nTransitions_r = nTransitions_c; //TODO different dataset

        arma::vec theta_c(phi.n_cols,arma::fill::zeros);
        arma::vec margin(nTransitions_c*nActions,arma::fill::ones);

        arma::mat phi_sample_c(nTransitions_c,phi.n_cols, arma::fill::zeros);
        arma::mat phi_sample_star_c(nTransitions_c,phi.n_cols, arma::fill::zeros);
        arma::mat phi_sample_star_r(nTransitions_r,phi.n_cols, arma::fill::zeros);

        unsigned int counter = 0;

        //definition of margin function
        unsigned int index = 0;
        for (auto& episode : data)
        {
        	for(unsigned t = 0; t + 1 < episode.size(); t++)
        	{
        		auto& tr = episode[t];
        		 margin(index+nTransitions_c*tr.u) = 0;
        		 phi_sample_c.row(index) = phi.row(index+nTransitions_c*tr.u);
        	}
        }

        double criterion = epsilon+1;
        double delta = 1.0/(counter+1);

        //Gradient descend
        while(criterion > epsilon && counter < N_final)
        {

            arma::mat Q_classif=phi*theta_c+margin;
            arma::uvec uMax;
            getMaxQIndex(Q_classif,uMax);
            for(unsigned int i=0; i < nTransitions_c; i++)
                phi_sample_star_c.row(i) = phi.row(i+nTransitions_c*uMax(i));

            arma::mat derivative_theta_c_c = arma::ones(1,nTransitions_c)*
                                             (phi_sample_star_c-phi_sample_c)/phi.n_rows;


            arma::mat Q_sample = phi_r*theta_c;
            arma::mat Q_next = phi_r_next*theta_c;

            arma::vec Vmax;
            arma::uvec vector;
            max_Q_v2(Q_next, Vmax, vector);

            for (unsigned int i=0; i < nTransitions_r; i++)
                phi_sample_star_r.row(i) = phi_r_next.row(vector(i));
            arma::mat derivative_theta_c_r = arma::sign(Q_sample-gamma*Vmax).t()*
                                             (phi_r-gamma*phi_sample_star_r)*lambda_RCAL/nTransitions_r;

            arma::mat derivative_theta_c = derivative_theta_c_c+derivative_theta_c_r;
            arma::vec oldTheta_c = theta_c;

            //update theta_r and theta_c vectors
            if (norm(derivative_theta_c,2)!=0)
                theta_c=theta_c-delta/(norm(derivative_theta_c,2))*derivative_theta_c.t();
            counter++;
            delta=1.0/counter;

            criterion = norm(theta_c-oldTheta_c);
        }

        return theta_c;

    }


    void getMaxQIndex(const arma::mat& Q, arma::uvec& uMax)
    {
        unsigned int N = Q.n_rows/nActions;
        uMax.set_size(N);

        for (unsigned int i = 0; i < N; i++)
        {
            double max = -std::numeric_limits<double>::infinity();

            for(unsigned int u = 0; u < nActions; u++)
            {
                double q = Q(i+N*u);
                if(q > max)
                {
                    max = q;
                    uMax(i) = u;
                }
            }
        }

    }

    void max_Q_v2(arma::mat Q, arma::vec& V, arma::uvec& vector)
    {
		unsigned int N = Q.n_rows/nActions;
		arma::mat Q_buffer(N,nActions, arma::fill::zeros);
		vector.zeros(N);
		for (unsigned int i=0; i < N; i++)
		{
			double qMax = -std::numeric_limits<double>::infinity();
			unsigned int uMax = 0;

			for(unsigned int u=0; u < nActions; u++)
			{
				double q = Q(i+N*u);
				if(q > qMax)
				{
					qMax = q;
				    uMax = u;
				}
			}

			V(i) = qMax;
			vector(i)=i+N*uMax;
		}
    }

    void regression(const arma::mat& phi, const arma::mat& y)
    {
        arma::mat A = phi.t()*phi+lambda_r*arma::eye(phi.n_cols, phi.n_cols);

        arma::vec b = phi.t()*y;

        arma::vec w = arma::solve(A, b);

        rewardf.setParameters(w);
    }


private:
    Dataset<FiniteAction, StateC>& data;
    LinearApproximator& rewardf;
    double gamma;
    unsigned int nActions;
    bool heuristic;

    double lambda_r;
    double lambda_RCAL;
    double epsilon;
    unsigned int N_final;

};

}


#endif /* INCLUDE_RELE_IRL_ALGORITHMS_CSI_H_ */
