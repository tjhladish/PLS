#ifndef PLS_H
#define PLS_H

#include <Eigen/Eigenvalues>
#include <vector>
#include <iostream>

#ifdef MPREAL_SUPPORT
#include "mpreal.h"
#include <unsupported/Eigen/MPRealSupport>
    using namespace mpfr;
    typedef mpreal float_type;
#else
    typedef double float_type;
#endif

#include <algorithm> // sort
#include <numeric> // iota

// tag/index sort, from https://stackoverflow.com/a/37732329/167973
template<typename T>
std::vector<size_t> ordered(const T& v) {
    std::vector<size_t> result(v.size());
    std::iota(std::begin(result), std::end(result), 0);
    std::sort(
        std::begin(result), std::end(result),
        [&v](const auto & lhs, const auto & rhs) {
            return *(v.begin() + lhs) < *(v.begin()+ rhs);
        }
    );
    return result;
}


//using namespace std;
using namespace Eigen;

using std::complex;

typedef Matrix<float_type, Dynamic, Dynamic> Mat2D;
typedef Matrix<float_type, Dynamic, 1>  Col;
typedef Matrix<int, Dynamic, 1>  Coli;
typedef Matrix<size_t, Dynamic, 1>  Colsz;
typedef Matrix<float_type, 1, Dynamic>  Row;
typedef Matrix<int, 1, Dynamic>  Rowi;
typedef Matrix<size_t, 1, Dynamic>  Rowsz;
typedef Matrix<complex<float_type>, Dynamic, Dynamic> Mat2Dc;
typedef Matrix<complex<float_type>, Dynamic, 1>  Colc;

/*
 *   Variable definitions from source paper:
 *     X     : predictor variables matrix (N × K)
 *     Y     : response variables matrix (N × M)
 *     B_PLS : PLS regression coefficients matrix (K × M)
 *     W     : PLS weights matrix for X (K × A)
 *     P     : PLS loadings matrix for X (K × A)
 *     Q     : PLS loadings matrix for Y (M × A)
 *     R     : PLS weights matrix to compute scores T directly from original X (K × A)
 *     T     : PLS scores matrix of X (N × A)
 *     w_a   : a column vector of W
 *     p_a   : a column vector of P
 *     q_a   : a column vector of Q
 *     r_a   : a column vector of R
 *     t_a   : a column vector of T
 *     K     : number of X-variables
 *     M     : number of Y-variables
 *     N     : number of objects
 *     A     : number of components in PLS model
 *     a     : integer counter for latent variable dimension
 */

// helper methods
template<typename MATTYPE>
size_t find_dominant_ev(const EigenSolver<MATTYPE> es) {
    auto eig_val = es.eigenvalues();
    float_type m = 0;
    size_t idx = 0;

    for (size_t i = 0; i < static_cast<size_t>(eig_val.size()); i++) {
        if (imag(eig_val[i]) == 0) {
            if (abs(eig_val[i]) > m) {
                m = abs(eig_val[i]);
                idx = i;
            }
        }
    }
    return idx;

}

// converts an Eigen vector type (Row or Col) to a std::vector
template<typename EIGENTYPE>
std::vector<float_type> to_cvector(const EIGENTYPE & data) {
    std::vector<float_type> vec(data.size());
    for (size_t i = 0; i < static_cast<size_t>(data.size()); i++) vec[i] = data[i];
    return vec;
}

// converts a std::vector to an Eigen vector type (Row or Col)
template<typename EIGENTYPE>
inline EIGENTYPE to_evector(const std::vector<float_type> & data) {
    EIGENTYPE row(data.size());
    for (size_t i = 0; i < data.size(); i++) row[i] = data[i];
    return row;
}

namespace PLS {
    // input helper functions
    std::vector<std::string> split(const std::string & s, const char separator = ',');
    Mat2D read_matrix_file(const std::string & filename, const char sep, const bool verbose = false, std::ostream & os = std::cerr);

    // PLS common functions
    Row colwise_stdev(const Mat2D & mat, const Row & means);
    Mat2D colwise_z_scores( const Mat2D & mat, const Row & mean, const Row & stdev);
    Mat2D colwise_z_scores( const Mat2D & mat );

    // PLS enumerated types
    typedef enum { KERNEL_TYPE1, KERNEL_TYPE2 } METHOD;
    typedef enum { PRESS, RMSEP } VALIDATION_OUTPUT;
    typedef enum { LOO, LSO, NEW_DATA } VALIDATION_METHOD;

}

struct PLS_Model {

    using namespace PLS;

    PLS_Model(
      const size_t num_predictors, const size_t num_responses, const size_t num_components
    ) : A(num_components) {
        P.setZero(num_predictors, A);
        W.setZero(num_predictors, A);
        R.setZero(num_predictors, A);
        Q.setZero(num_responses, A);
        // T will be initialized if needed
    }

    PLS_Model(
        const Mat2D& X, const Mat2D& Y,
        const size_t num_components,
        const METHOD algorithm = KERNEL_TYPE1
    ) : PLS_Model(X.cols(), Y.cols(), num_components) {
        method = algorithm;
        plsr(X, Y, algorithm);
    }

    PLS_Model& plsr (const Mat2D& X, const Mat2D& Y, const METHOD algorithm);

    // latent X values, i.e. the orthogonal metrics you wish you could measure
    const Mat2Dc scores(const Mat2D& X_new, const size_t comp) const;
    const Mat2Dc scores(const Mat2D& X_new) const { return scores(X_new, A); }

    const Mat2Dc loadingsX(const size_t comp) const;
    const Mat2Dc loadingsX() const { return loadingsX(A); }

    const Mat2Dc loadingsY(const size_t comp) const; 
    const Mat2Dc loadingsY() const { return loadingsY(A); }

    // compute the regression coefficients (aka 'beta')
    const Mat2Dc coefficients(const size_t comp) const;
    const Mat2Dc coefficients() const { return coefficients(A); }


    // predicted Y values, given X values and pls model
    const Mat2D fitted_values(const Mat2D& X, const size_t comp) const;
    const Mat2D fitted_values(const Mat2D& X) const { return fitted_values(X, A); }

    // unexplained portion of Y values
    const Mat2D residuals(const Mat2D& X, const Mat2D& Y, const size_t comp) const;
    const Mat2D residuals(const Mat2D& X, const Mat2D& Y) const { return residuals(X, Y, A); }

    // Sum of squared errors
    const Row SSE(const Mat2D& X, const Mat2D& Y, const size_t comp) const;
    const Row SSE(const Mat2D& X, const Mat2D& Y) const { return SSE(X, Y, A); }

    // Total sum of squares
    const Row SST(const Mat2D& Y) const;

    // fraction of explainable variance
    const Row explained_variance(const Mat2D& X, const Mat2D& Y, const size_t comp) const;
    const Row explained_variance(const Mat2D& X, const Mat2D& Y) const { return explained_variance(X, Y, A); }

    // leave-one-out validation of model (i.e., are we overfitting?)
    Mat2D loo_validation(const Mat2D& X, const Mat2D& Y, const VALIDATION_OUTPUT out_type) const;
    Mat2D lso_validation(const Mat2D& X, const Mat2D& Y, const VALIDATION_OUTPUT out_type, const float_type test_fraction, const size_t num_trials);

    std::vector<Mat2D> _loo_cv_error_matrix(const Mat2D& X, const Mat2D& Y) const;
    std::vector<Mat2D> _lso_cv_error_matrix(const Mat2D& X, const Mat2D& Y, const float_type test_fraction, const size_t num_trials) const;
    std::vector<Mat2D> _new_data_cv_error_matrix(const Mat2D& X_new, const Mat2D& Y_new) const;

    // if val_method is LOO, X and Y should be original data
    // if val_method is NEW_DATA, X and Y should be observations not included in the original model
    const Rowsz optimal_num_components(
        const Mat2D& X, const Mat2D& Y, const VALIDATION_METHOD val_method,
        const float_type test_fraction = 0, const size_t num_trials = 0,
        mt19937 & rng // TODO pass a null here by default
    ) const;

    // output methods
    void print_explained_variance(const Mat2D& X, const Mat2D& Y, std::ostream& os = std::cerr) const;
    void print_state(std::ostream& os) const;
    void print_model_assessment(
        const Mat2D & X, const Mat2D & Y,
        const size_t training_size, const size_t testing_size,
        const size_t optimal_components, const size_t used_components,
        std::ostream& os = std::cerr
    ) const;

    void print_validation(
        const Mat2D & X, const Mat2D & Y,
        const size_t training_size, const size_t testing_size,
        const size_t optimal_components, const size_t used_components,
        std::ostream& os
    ) const;

    private:
        size_t A; // number of components
        Mat2Dc P, W, R, Q, T;
        METHOD method;

};


#endif
