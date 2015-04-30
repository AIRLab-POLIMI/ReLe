#ifndef INTERFACES_H_
#define INTERFACES_H_

#include <cstring>
#include <fstream>

namespace ReLe
{

class WritableInterface
{
public:
    virtual ~WritableInterface()
    { }

    virtual void persist(const std::string& f) const {}
    virtual void resurrect(const std::string& f) {}

    /**
     * Write a complete description of the instance to
     * a stream.
     * @brief Insert formatted output into stream
     * @param out the output stream
     */
    virtual void writeOnStream (std::ostream& out) = 0;

    /**
     * Read the description of the basis function from
     * a file and reset the internal state according to that.
     * This function is complementary to WriteOnStream
     * @brief Extract formatted input from stream
     * @param in the input stream
     */
    virtual void readFromStream (std::istream& in) = 0;
};

} //end namespace


#endif //INTERFACES_H_
