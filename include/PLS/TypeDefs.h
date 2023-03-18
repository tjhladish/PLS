
#ifndef PLS_TYPEDEFS_H
#define PLS_TYPEDEFS_H

#include <Eigen/Core> // only need to know that Matrixes are a thing

#ifdef MPREAL_SUPPORT
#include "mpreal.h"
#include <unsupported/Eigen/MPRealSupport>
    using namespace mpfr;
    typedef mpreal float_type;
#else
    typedef double float_type;
#endif // MPREAL_SUPPORT

typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic> Mat2D;
typedef Eigen::Matrix<float_type, Eigen::Dynamic, 1> Col;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> Coli;
typedef Eigen::Matrix<size_t, Eigen::Dynamic, 1> Colsz;
typedef Eigen::Matrix<float_type, 1, Eigen::Dynamic> Row;
typedef Eigen::Matrix<int, 1, Eigen::Dynamic> Rowi;
typedef Eigen::Matrix<size_t, 1, Eigen::Dynamic> Rowsz;
typedef Eigen::Matrix<std::complex<float_type>, Eigen::Dynamic, Eigen::Dynamic> Mat2Dc;
typedef Eigen::Matrix<std::complex<float_type>, Eigen::Dynamic, 1> Colc;
typedef std::vector<Mat2D> PLSError;

#endif // PLS_TYPEDEFS_H