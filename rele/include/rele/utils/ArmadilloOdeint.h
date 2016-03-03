#ifndef ARMADILLOODEINT_H_
#define ARMADILLOODEINT_H_

#include <armadillo>
#include <boost/numeric/odeint.hpp>

// define arma::vec as resizeable

namespace boost
{
namespace numeric
{
namespace odeint
{

template<>
struct is_resizeable<arma::vec>
{
    typedef boost::true_type type;
    static const bool value = type::value;
};

template<>
struct same_size_impl<arma::vec, arma::vec>
{
    // define how to check size
    static bool same_size(const arma::vec &v1,
                          const arma::vec &v2)
    {
        return v1.n_elem == v2.n_elem;
    }
};

template<>
struct resize_impl<arma::vec, arma::vec>
{
    // define how to resize
    static void resize(arma::vec &v1,
                       const arma::vec &v2)
    {
        v1.resize(v2.n_elem);
    }
};

}//end namespace odeint
}//end namespace numeric
}//end namespace boost


#endif //ARMADILLOODEINT_H_

