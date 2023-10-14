
#ifndef PLS_TYPES_H
#define PLS_TYPES_H

#include <complex>

#ifdef MPREAL_SUPPORT
#include "mpreal.h"
    typedef mpfr::mpreal float_type;
#else
    typedef double float_type;
#endif // MPREAL_SUPPORT

// forward declaration of all the Eigen types
// in dependent code, include this in header files (to forward declare the types)
// and include <Eigen/Core> in object source files to get e.g. Matrix accessor methods
namespace Eigen {

    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    class Matrix;
// TODO get this from cmake EIGEN_DEFAULT_DENSE_INDEX_TYPE
    typedef std::ptrdiff_t Index;
}

using Mat2D = Eigen::Matrix<float_type, -1, -1, 0, -1, -1>;
using Mat2Dc = Eigen::Matrix<std::complex<float_type>, -1, -1, 0, -1, -1>;

using Col = Eigen::Matrix<float_type, -1, 1, 0, -1, 1>;
using Colc = Eigen::Matrix<std::complex<float_type>, -1, 1, 0, -1, 1>;
using Coli = Eigen::Matrix<int, -1, 1, 0, -1, 1>;
using Colsz = Eigen::Matrix<size_t, -1, 1, 0, -1, 1>;

// n.b. rows must be _Options = 1, i.e. RowMajor
using Row = Eigen::Matrix<float_type, 1, -1, 1, 1, -1>;
using Rowi = Eigen::Matrix<int, 1, -1, 1, 1, -1>;
using Rowsz = Eigen::Matrix<size_t, 1, -1, 1, 1, -1>;

#endif // PLS_TYPES_H