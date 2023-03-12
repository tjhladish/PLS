#ifndef PLS_H
#define PLS_H

#include <Eigen/Eigenvalues>
#include <vector>
#include <iostream> // for ostream
#include <random> // for std::mt19937
#include <algorithm> // sort
#include <numeric> // iota

#ifdef MPREAL_SUPPORT
#include "mpreal.h"
#include <unsupported/Eigen/MPRealSupport>
    using namespace mpfr;
    typedef mpreal float_type;
#else
    typedef double float_type;
#endif

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
typedef std::vector<Mat2D> PLSError;

namespace PLS {

/* ------- Eigen-related convenience functions for PLS specific operations ------- */

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

    // converts an Eigen vector type (Row or Col) to a std::vector
    template<typename EIGENTYPE>
    std::vector<float_type> to_cvector(const EIGENTYPE & data) {
        std::vector<float_type> vec(data.begin(), data.end());
        return vec;
    }

    // converts a std::vector to an Eigen vector type (Row or Col)
    // TODO: should this be a Eigen::Map type (i.e. a Matrix-like view of the data)?
    template<typename EIGENTYPE>
    inline EIGENTYPE to_evector(const std::vector<float_type> & data) {
        EIGENTYPE row(data.size());
        for (size_t i = 0; i < data.size(); i++) row[i] = data[i];
        return row;
    }


    // for splitting lines from plaintext input file
    std::vector<std::string> split(const std::string & s, const char separator = ',');

    // for reading a matrix from a plaintext file; assumes no header, each line is a row
    // will exit if any rows have different number of columns, or on std::stod errors
    Mat2D read_matrix_file(const std::string & filename, const char separator = ',');

    // for calculating the Total Sum of Squares (TSS) by column (mean specified)
    // TSS = sum((x - mean)^2)
    Row SST(const Mat2D & mat, const Row & means);
    // for calculating the Total Sum of Squares (TSS) by column (mean calculated)
    Row SST(const Mat2D & mat);

    // for calculating the stdev by column (mean specified)
    Row colwise_stdev(const Mat2D & mat, const Row & means);
    // for calculating the stdev by column (mean calculated)
    Row colwise_stdev(const Mat2D & mat);

    // for converting a matrix to z-scores by column (mean, stdev specified)
    Mat2D colwise_z_scores(const Mat2D & mat, const Row & mean, const Row & stdev);
    // for converting a matrix to z-scores by column (mean, stdev calculated)
    Mat2D colwise_z_scores(const Mat2D & mat);

    // given an EigenSolver object, returns the index of the dominant eigenvalue
    // (used internally in dominant_eigen(value|vector))
    template<typename MATTYPE>
    size_t find_dominant_ev(const EigenSolver<MATTYPE> & es);

    // returns the dominant (real-valued) eigenvalue
    float_type dominant_eigenvalue(const EigenSolver<Mat2Dc> & es);

    // returns the dominant eigenvector (possibly complex-valued)
    Colc dominant_eigenvector(const EigenSolver<Mat2D> & es);

    // empirical approximation of the normalCDF
    float_type normalcdf(const float_type z);

    // computes the Wilcoxon signed-rank test statistic
    float_type wilcoxon(const Col & err_1, const Col & err_2);

    // given a full set of indices, shuffles it, and places a k-sized and n-k-sized
    // sample into the sample and complement vectors, respectively
    void rand_nchoosek(
        std::mt19937 & rng,
        std::vector<Eigen::Index> & full,
        std::vector<Eigen::Index> & sample,
        std::vector<Eigen::Index> & complement
    );

/* -------------------- PLS class/analysis related definitions -------------------- */

    // PLS kernal enumerated types
    typedef enum { KERNEL_TYPE1, KERNEL_TYPE2 } METHOD;
    
    // Validation output types - used in `validation()` function
    //  - residual error sum of squares (RESS); 
    //  - mean square error (MSE);
    // whether these are CV (cross-validation) or P (prediction) depends
    // on what they were calculated from, e.g. new or holdout data VS
    // some partition of the original data
    // To obtain the R (root) versions, follow with `cwiseSqrt()` (coefficient-wise square root).
    // The difference between RESS and MSE: MSE divides by N, the number
    // of observations in the error argument. Thus, MSE can be recovered from RESS
    // result, by dividing by error[0].rows()
    typedef enum { RESS, MSE } VALIDATION_OUTPUT;

    // Validation methods - used in `error()` functions
    // leave-one-out (LOO) validation;
    // leave-some-out (LSO) validation;
    // new data (NEW_DATA) validation
    typedef enum { LOO, LSO, NEW_DATA } VALIDATION_METHOD;

    Mat2D validation(
        const PLSError & errors,
        const PLS::VALIDATION_OUTPUT out_type
    );

    Colsz optimal_num_components(
        const PLSError & errors,
        const float_type ALPHA = 0.1
    );

    void print_validation(
        const PLSError & errors,
        const PLS::VALIDATION_METHOD method,
        const PLS::VALIDATION_OUTPUT out_type,
        std::ostream & os = std::cerr
    );

}

/*
 *   PLS regression object
 *   --------------------------------
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
struct PLS_Model {

    PLS_Model & plsr (const Mat2D& X, const Mat2D& Y, const PLS::METHOD algorithm);

    // use when expecting to re-apply plsr repeatedly to new data of the same shape
    PLS_Model(
      const size_t num_predictors, const size_t num_responses, const size_t num_components
    ) : A(num_components) {
        P.setZero(num_predictors, A);
        W.setZero(num_predictors, A);
        R.setZero(num_predictors, A);
        Q.setZero(num_responses, A);
        // T will be initialized if needed
    }

    // use for a one-off PLSR
    PLS_Model(
        const Mat2D& X, const Mat2D& Y,
        const size_t num_components,
        const PLS::METHOD algorithm = PLS::KERNEL_TYPE1
    ) : PLS_Model(X.cols(), Y.cols(), num_components) {
        method = algorithm;
        plsr(X, Y, algorithm);
    }

    // Transforms X_new into the latent space of the PLS model
    // i.e. the orthogonal X you wish you could measure
    const Mat2Dc scores(const Mat2D& X_new, const size_t comp) const;
    // default to originally-specified number of components
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

    // Sum of squared errors: uses estimated PLS model on X to predict Y, computes
    // the difference between the predicted and actual Y values, and squares the residual
    // and then sums (by column, the Y components) over all observations
    const Row SSE(const Mat2D& X, const Mat2D& Y, const size_t comp) const;
    const Row SSE(const Mat2D& X, const Mat2D& Y) const { return SSE(X, Y, A); }

    // fraction of explainable variance
    const Row explained_variance(const Mat2D& X, const Mat2D& Y, const size_t comp) const;
    const Row explained_variance(const Mat2D& X, const Mat2D& Y) const { return explained_variance(X, Y, A); }

    template <PLS::VALIDATION_METHOD val_method>
    PLSError error(
        const Mat2D& X, const Mat2D& Y
    ) const {
        std::cerr << "error<" << val_method <<"> must be provided additional arguments." << std::endl;
        exit(-1);
    }

    template <PLS::VALIDATION_METHOD val_method>
    PLSError error(
        const Mat2D& X, const Mat2D& Y,
        const float_type test_fraction, const size_t num_trials, std::mt19937 & rng
    ) const {
        std::cerr << "error<" << val_method <<"> provided too many arguments." << std::endl;
        exit(-1);
    }

    // output methods
    void print_explained_variance(
        const Mat2D& X, const Mat2D& Y, std::ostream& os = std::cerr
    ) const;

    void print_state(std::ostream& os = std::cerr) const;

    private:
        size_t A; // number of components
        Mat2Dc P, W, R, Q, T;
        PLS::METHOD method;

};


#endif
