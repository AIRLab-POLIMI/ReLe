/*
 * DoubleFQI.h
 *
 *  Created on: Dec 8, 2015
 *      Author: shirokuma
 */

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_DOUBLEFQI_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_DOUBLEFQI_H_

#include "FQI.h"


namespace ReLe
{

template<class StateC>
class DoubleFQI: public FQI<StateC>
{
public:

    /* This class implements the Double FQI algorithm. As a batch algorithm, it takes
     * a dataset of (s, a, r, s') transitions, together with the regressors that
     * it is used to approximate the target distribution of Q values.
     */
    DoubleFQI(Dataset<FiniteAction, StateC>& data, BatchRegressor& QRegressorA,
              BatchRegressor& QRegressorB, unsigned int nStates, unsigned int nActions, double gamma) :
        FQI<StateC>(data, QRegressorA, nStates, nActions, gamma),
        QRegressorA(QRegressorA),
        QRegressorB(QRegressorB)
    {
    }

    void run(Features& phi, unsigned int maxiterations, double epsilon) override
    {
        /* This is the function to be called to run the Double FQI algorithm. It takes
         * the features phi that are used to compute the input of the regressor.
         */

        // Compute the overall number of samples
        unsigned int nEpisodes = this->data.size();
        unsigned int nSamples = 0;
        for(unsigned int k = 0; k < nEpisodes; k++)
            nSamples += this->data[k].size();

        // Rewards are extracted from the dataset
        arma::mat rewards = this->data.rewardAsMatrix();
        // Input of the regressors are computed using provided features
        arma::mat input = this->data.featuresAsMatrix(phi);
        /* Output vector is initialized. It will contain the Q values, found
         * with the optimal Bellman equation, that are used in regression as
         * target values.
         */
        arma::mat output(1, input.n_cols, arma::fill::zeros);

        // This vector is used for the terminal condition evaluation
        this->QHat.zeros(input.n_cols);

        /* First iteration of Double FQI is performed here training
         * the regressor with a dataset that has the rewards as
         * output.
         */
        BatchDataFeatures<arma::vec, arma::vec> featureDatasetStart(input, rewards);
        // Modo per inizializzare i regressori (cercare modo migliore)
        BatchDataFeatures<arma::vec, arma::vec> featureDatasetForInitialization(input, output);

        unsigned int selectedRegressor = RandomGenerator::sampleUniformInt(0, 1);
        if(selectedRegressor == 0)
        {
        	QRegressorA.trainFeatures(featureDatasetStart);
        	QRegressorB.trainFeatures(featureDatasetForInitialization);
        }
        else
        {
        	QRegressorB.trainFeatures(featureDatasetStart);
        	QRegressorA.trainFeatures(featureDatasetForInitialization);
        }

        // Initial QHat is stored before update
        arma::vec prevQHat = this->QHat;

        // Update QHat
        this->computeQHat(this->data);

        unsigned int iteration = 0;
        double J;
        // Print info
        std::cout << std::endl << "*********************************************************" << std::endl;
        std::cout << "Double FQI iteration: " << iteration << std::endl;
        std::cout << "*********************************************************" << std::endl;
        this->printInfo(output, prevQHat, J);

        // Main FQI loop
        while(iteration < maxiterations && J > epsilon)
        {
            // Update and print the iteration number
            iteration++;
            std::cout << std::endl << "*********************************************************" << std::endl;
            std::cout << "Double FQI iteration: " << iteration << std::endl;
            std::cout << "*********************************************************" << std::endl;

            // Choose next regressor to train
            selectedRegressor = RandomGenerator::sampleUniformInt(0, 1);
            if(selectedRegressor == 0)
            	doubleFQIStep(QRegressorA, QRegressorB, input, output, rewards);
            else
            	doubleFQIStep(QRegressorB, QRegressorA, input, output, rewards);

            // Previous Q approximated values are stored
            prevQHat = this->QHat;
            /* New Q values are computed using the regressor trained with the
             * new output values.
             */
            this->computeQHat(this->data);

            // Print info
            this->printInfo(output, prevQHat, J);
        }

        // Print final info
        std::cout << "*********************************************************" << std::endl;
        if(J > epsilon)
            /* The algorithm has not converged and terminated for exceeding
             * the maximum number of transitions.
             */
            std::cout << "FQI finished in " << iteration <<
                      " iterations WITHOUT CONVERGENCE to a fixed point" << std::endl;
        else
            // Error below the maximum desired error: the algorithm has converged
            std::cout << "FQI converged in " << iteration << " iterations" << std::endl <<
                      "********************************************************* " << std::endl;

        this->printPolicy();
    }

    void doubleFQIStep(BatchRegressor& trainingRegressor,
    				   BatchRegressor& evaluationRegressor,
					   const arma::mat input,
					   arma::mat& output,
					   const arma::mat rewards)
    {
        // Loop on each dataset sample (i.e. on each transition)
        unsigned int i = 0;

        for(auto& episode : this->data)
        {
            for(auto& tr : episode)
            {
                /* For the current s', Q values for each action are stored in
                 * Q_xn. The optimal Bellman equation can be computed
                 * finding the maximum value inside Q_xn. They are zero if
                 * xn is an absorbing state. Note that here we exchange the
                 * regressor according to Double Q-Learning algorithm.
                 */
                arma::vec Q_xn(this->nActions, arma::fill::zeros);
                if(!tr.xn.isAbsorbing())
                    for(unsigned int u = 0; u < this->nActions; u++)
                        Q_xn(u) = arma::as_scalar(trainingRegressor(tr.xn, FiniteAction(u)));

                /* For the current s', Q values for each action are stored in
                 * Q_xn. The optimal Bellman equation can be computed
                 * finding the maximum value inside Q_xn. They are zero if
                 * xn is an absorbing state.
                 */
                double qmax = Q_xn.max();
                arma::uvec maxIndex = arma::find(Q_xn == qmax);
                unsigned int index = RandomGenerator::sampleUniformInt(0,
                                     maxIndex.n_elem - 1);
                output(i) = arma::as_scalar(rewards(0, i) + this->gamma * evaluationRegressor(tr.xn, FiniteAction(index)));

                i++;
            }
        }

        // The regressor is trained
        BatchDataFeatures<arma::vec, arma::vec> featureDataset(input, output);
        trainingRegressor.trainFeatures(featureDataset);
    }

    // Vedere se si riesce ad usare il regressore in FQI come "regressore medio di QA e QB"
    // In tal caso si toglierebbero queste due funzioni
    void computeQHat(Dataset<FiniteAction, StateC>& data) override
    {
        // Computation of Q values approximation with the updated regressor
        unsigned int i = 0;
        for(auto& episode : data)
        {
            for(auto& tr : episode)
            {
                this->QHat(i) = arma::as_scalar(QRegressorA(tr.x, tr.u) + QRegressorB(tr.x, tr.u) / 2);
                i++;
            }
        }
    }

    void computeQTable() override
    {
    	for(unsigned int i = 0; i < this->nStates; i++)
    		for(unsigned int j = 0; j < this->nActions; j++)
    			this->QTable(i, j) = arma::as_scalar(
    					QRegressorA(FiniteState(i), FiniteAction(j)) + QRegressorB(FiniteState(i), FiniteAction(j)) / 2);
    }

protected:
    BatchRegressor& QRegressorA;
    BatchRegressor& QRegressorB;
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_DOUBLEFQI_H_ */
