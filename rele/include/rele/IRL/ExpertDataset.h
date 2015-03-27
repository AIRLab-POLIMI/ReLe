/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef INCLUDE_RELE_IRL_EXPERTDATASET_H_
#define INCLUDE_RELE_IRL_EXPERTDATASET_H_

namespace ReLe
{

template<class ActionC, class StateC>
class ExpertDataset
{
public:
    ExpertDataset()
	{

	}

    arma::mat computefeatureExpectation(AbstractBasisMatrix& basis, double gamma = 1)
    {
    	size_t episodes = data.size();
    	arma::mat featureExpectation(basis.rows(), basis.cols(), arma::fill::zeros);

    	for(auto& episode : data)
    	{
    		arma::mat episodefeatureExpectation(basis.rows(), basis.cols(), arma::fill::zeros);;

    		for(unsigned int t = 0; t < episode.size(); t++)
    		{
    			Transition<ActionC, StateC>& transition = episode[t];
    			episodefeatureExpectation += std::pow(gamma, t) * basis(transition.x);
    		}

    		Transition<ActionC, StateC>& transition = episode.back();
    		episodefeatureExpectation += std::pow(gamma, episode.size() + 1) * basis(transition.xn);


    		featureExpectation += episodefeatureExpectation;
    	}

    	featureExpectation /= episodes;

    	return featureExpectation;
    }

    void addData(TrajectoryData<ActionC, StateC>& data)
    {
    	this->data.insert(this->data.end(), data.begin(), data.end());
    }

    void setData(TrajectoryData<ActionC, StateC>& data)
    {
    	this->data(data);
    }

private:
    TrajectoryData<ActionC, StateC> data;

};

}


#endif /* INCLUDE_RELE_IRL_EXPERTDATASET_H_ */
