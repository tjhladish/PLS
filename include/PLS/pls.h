#ifndef PLS_H
#define PLS_H

#include <PLS/types.h> // only need to know forward declarations for header

#include <vector>
#include <iostream> // for ostream, std::cerr
#include <random> // for std::mt19937
#include <algorithm> // sort
#include <numeric> // iota

/**
@brief Namespace for all Partial Least Square (PLS) related functions and classes.
*/
namespace PLS {

    // error data structure
    typedef std::vector<Mat2D> ResidualData;
    class Model; // forward declaration for Residual friendship

    class Residual {
        const std::vector<Mat2D> _residual;
        const std::string _method_label;
        Residual(const std::vector<Mat2D> &residual, const std::string &method) : _residual(residual), _method_label(method) {}
        friend class Model;

        public:            
            const std::vector<Mat2D> errors() const { return _residual; }
            const std::string method() const { return _method_label; }
    };

/* ------- Eigen-related convenience functions for PLS specific operations ------- */

    // tag/index sort, from https://stackoverflow.com/a/37732329/167973
    template<typename T>
    std::vector<size_t> ordered(const T &v) {
        std::vector<size_t> result(v.size());
        std::iota(std::begin(result), std::end(result), 0);
        std::sort(
            std::begin(result), std::end(result),
            [&v](const auto &lhs, const auto &rhs) {
                return *(v.begin() + lhs) < *(v.begin()+ rhs);
            }
        );
        return result;
    }

    // converts an Eigen vector type (Row or Col) to a std::vector
    template<typename EIGENTYPE>
    std::vector<float_type> to_cvector(const EIGENTYPE &data) {
        std::vector<float_type> vec(data.begin(), data.end());
        return vec;
    }

    // converts a std::vector to an Eigen vector type (Row or Col)
    // TODO: should this be a Eigen::Map type (i.e. a Matrix-like view of the data)?
    template<typename EIGENTYPE>
    inline EIGENTYPE to_evector(const std::vector<float_type> &data) {
        EIGENTYPE v(data.size());
        for (size_t i = 0; i < data.size(); i++) { v[i] = data[i]; }
        return v;
    }


    // for splitting lines from plaintext input file
    std::vector<std::string> split(const std::string &s, const char separator = ',');

    // for reading a matrix from a plaintext file; assumes no header, each line is a row
    // will exit if any rows have different number of columns, or on std::stod errors
    Mat2D read_matrix_file(const std::string &filename, const char separator = ',');

    // for calculating the Total Sum of Squares (TSS) by column (mean specified)
    // TSS = sum((x - mean)^2)
    Row SST(const Mat2D &mat, const Row &means);
    // for calculating the Total Sum of Squares (TSS) by column (mean calculated)
    Row SST(const Mat2D &mat);

    // for calculating the stdev by column (mean specified)
    Row colwise_stdev(const Mat2D &mat, const Row &means);
    // for calculating the stdev by column (mean calculated)
    Row colwise_stdev(const Mat2D &mat);

    Row z_scores(const Row &obs, const Row &mean, const Row &stdev);

    // for converting a matrix to z-scores by column (mean, stdev specified)
    Mat2D colwise_z_scores(const Mat2D &mat, const Row &mean, const Row &stdev);
    // for converting a matrix to z-scores by column (mean, stdev calculated)
    Mat2D colwise_z_scores(const Mat2D &mat);

    // empirical approximation of the normalCDF
    float_type normalcdf(const float_type z);

    // computes the Wilcoxon signed-rank test statistic
    float_type wilcoxon(const Col &err_1, const Col &err_2);

    // given a full set of indices, shuffles it, and places a k-sized and n-k-sized
    // sample into the sample and complement vectors, respectively
    void rand_nchoosek(
        std::mt19937 &rng,
        std::vector<Eigen::Index> &full,
        std::vector<Eigen::Index> &sample,
        std::vector<Eigen::Index> &complement
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

    Mat2D validation(
        const Residual &residual,
        const PLS::VALIDATION_OUTPUT out_type
    );

    Colsz optimal_num_components(
        const Residual &residual,
        const float_type ALPHA = 0.1
    );

    void print_validation(
        const Residual &residual,
        const VALIDATION_OUTPUT out_type,
        std::ostream &os = std::cerr
    );

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
struct Model {

    // TODO: private? should generally only be used for cross-validation exercises
    Model(
        const Mat2D &X, const Mat2D &Y,
        const METHOD &algorithm,
        const size_t &max_components
    );

    Model(
        const Mat2D &X, const Mat2D &Y,
        const METHOD &algorithm = KERNEL_TYPE1
    );

    // TODO: private? should only be used for cross-validation exercises
    void plsr(const Mat2D &X, const Mat2D &Y, const METHOD &algorithm);

    // the modes below all default to originally-specified number of components

    // Transforms X_new into the latent space of the PLS model
    // i.e. the orthogonal X you wish you could measure
    const Mat2Dc scores(const Mat2D &X_new, const size_t comp) const;
    const Mat2Dc scores(const Mat2D &X_new) const;

    const Mat2Dc loadingsX(const size_t comp) const;
    const Mat2Dc loadingsX() const;
    const Mat2Dc loadingsY(const size_t comp) const; 
    const Mat2Dc loadingsY() const;

    // compute the regression coefficients (aka 'beta')
    const Mat2Dc coefficients(const size_t comp) const;
    const Mat2Dc coefficients() const;

    // predicted Y values, given X values and pls model
    const Mat2D fitted_values(const Mat2D &X, const size_t comp) const;
    const Mat2D fitted_values(const Mat2D &X) const;

    // unexplained portion of Y values
    const Mat2D residuals(const Mat2D &X, const Mat2D &Y, const size_t comp) const;
    const Mat2D residuals(const Mat2D &X, const Mat2D &Y) const;

    // Sum of squared errors: uses estimated PLS model on X to predict Y, computes
    // the difference between the predicted and actual Y values, and squares the residual
    // and then sums (by column, the Y components) over all observations
    const Row SSE(const Mat2D &X, const Mat2D &Y, const size_t comp) const;
    const Row SSE(const Mat2D &X, const Mat2D &Y) const;

    // fraction of explainable variance
    const Row explained_variance(const Mat2D &X, const Mat2D &Y, const size_t comp) const;
    const Row explained_variance(const Mat2D &X, const Mat2D &Y) const;

    // cross-validation methods:
    //  - LOO: leave-one-out => requires no new data
    //  - LSO: leave-some-out => requires no new data, but does need the subset sizes and random number generator
    //  - NEW_DATA: validating against new data - expects data outside the fitting data
    Residual cv_LOO() const;
    Residual cv_NEW_DATA(const Mat2D &X, const Mat2D &Y) const;
    Residual cv_LSO(const float_type test_fraction, const size_t num_trials, std::mt19937 &rng) const;

    // output methods
    void print_explained_variance(
        const Mat2D &X, const Mat2D &Y, std::ostream &os = std::cerr
    ) const;

    void print_state(std::ostream &os = std::cerr) const;

    ~Model();

    private:
        const size_t A; // number of components
        // TYPE * const == the pointer is a const, not the TYPE it points to

        // hang onto these for certain cross-validation methods
        const Mat2D * const _X;
        const Mat2D * const _Y;

        Mat2Dc * const P;
        Mat2Dc * const W;
        Mat2Dc * const R;
        Mat2Dc * const Q;
        Mat2Dc * const T;

        const PLS::METHOD method;

        // internal constructor for cross-validation methods
        Model(
            const size_t &num_predictors, const size_t &num_responses,
            const METHOD &algorithm = KERNEL_TYPE1
        );

        Model(
            const size_t &num_predictors, const size_t &num_responses,
            const METHOD &algorithm, const size_t &max_components
        );

};

} // namespace PLS

#endif
