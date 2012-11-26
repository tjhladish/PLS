#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
//#include <cstdlib>
#include <sstream>
#include <vector>
//#include <algorithm>
#include <assert.h>
//#include <iterator>
#include <iomanip>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

#ifdef MPREAL_SUPPORT
#include "mpreal.h"
#include <unsupported/Eigen/MPRealSupport>
    using namespace mpfr;
    typedef mpreal float_type;
#else
    typedef long double float_type;
#endif 

typedef Matrix<float_type,Dynamic,Dynamic> Mat2D;
typedef Matrix<float_type, Dynamic, 1>  Col;
typedef Matrix<float_type, 1, Dynamic>  Row;
typedef Matrix<complex<float_type>,Dynamic,Dynamic> Mat2Dc;
typedef Matrix<complex<float_type>, Dynamic, 1>  Colc;

void split(const string& s, char c, vector<string>& v);

inline double string2double(const std::string& s){ std::istringstream i(s); double x = 0; i >> x; return x; }

inline Row col_means( Mat2D mat ) { return mat.colwise().sum() / mat.rows(); }

Mat2D read_matrix_file(string filename, char sep); 

Row col_stdev( Mat2D mat, Row means );

float_type dominant_eigenvalue( EigenSolver<Mat2Dc> es );

Colc dominant_eigenvector( EigenSolver<Mat2D> es );

Mat2D colwise_z_scores( const Mat2D& mat );
#endif
