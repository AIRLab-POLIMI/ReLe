#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"
#include "basis/GaussianRBF.h"

namespace ReLe
{

BasisFunctions::BasisFunctions()
{
}

BasisFunctions::~BasisFunctions()
{
    BasisFunctions::iterator it;
    for (it = this->begin(); it != this->end(); ++it)
    {
        delete *it;
    }
}

void
BasisFunctions::GeneratePolynomialBasisFunctions(
    unsigned int degree, unsigned int input_size)
{
    //  AddBasisFunction(new PolynomialFunction(0,0));
    std::vector<unsigned int> dim;
    for (unsigned int i = 0; i < input_size; i++)
    {
        dim.push_back(i);
    }
    for (unsigned int d = 0; d <= degree; d++)
    {
        std::vector<unsigned int> deg(input_size);
        deg[0] = d;
        GeneratePolynomialsPermutations(deg, dim);
        GeneratePolynomials(deg, dim, 1);
    }
    std::cout << size() << " polynomial basis functions added!" << std::endl;
}


void BasisFunctions::display(std::vector<unsigned int> v)
{
    for (vector<unsigned int >::iterator it = v.begin(); it != v.end(); ++it)
    {
        std::cout << *it;
    }
    std::cout << std::endl;
}

void BasisFunctions::GeneratePolynomialsPermutations(vector<unsigned int> deg,
        vector<unsigned int>& dim)
{
    std::sort(deg.begin(), deg.end());
    do
    {
        BasisFunction* pBF = new PolynomialFunction(dim, deg);
        this->push_back(pBF);
        //    display(deg);
    }
    while (next_permutation(deg.begin(), deg.end()));
}

void BasisFunctions::GeneratePolynomials(vector<unsigned int> deg,
        vector<unsigned int>& dim,
        unsigned int place)
{
    if (deg.size() > 1)
    {
        if (deg[0] > deg[1] && deg[place] < deg[place - 1] && deg[0] - deg[place] > 1)
        {
            std::vector<unsigned int> degree = deg;
            degree[0]--;
            degree[place]++;
            GeneratePolynomialsPermutations(degree, dim);
            GeneratePolynomials(degree, dim, place);
            if (place < deg.size() - 1)
            {
                GeneratePolynomials(degree, dim, place + 1);
            }
        }
    }
}

std::ostream& operator<< (std::ostream& out, BasisFunctions& bf)
{
    out << bf.size() << std::endl;
    for (unsigned int i = 0; i < bf.size(); i++)
    {
        bf[i]->WriteOnStream(out);
        out << std::endl;
    }
    return out;
}

std::istream& operator>> (std::istream& in, BasisFunctions& bf)
{
    unsigned int num_basis_functions;
    in >> num_basis_functions;
    for (unsigned int i = 0; i < num_basis_functions; i++)
    {
        std::string type;
        in >> type;
        BasisFunction* function = 0;
        if (type == "Polynomial")
        {
            function = new PolynomialFunction();
        }
        else if (type == "GaussianRbf")
        {
            function = new GaussianRbf();
        }
        else
        {
            std::cerr << "ERROR: Unrecognized basis-function type" << std::endl;
            exit(1);
        }
//        in >> *function;
        function->ReadFromStream(in);
        bf.push_back(function);
    }
    return in;
}

}//end namespace
