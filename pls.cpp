#include "pls.h"
#include <cmath> // log10, ceil
#include <iomanip> // setw
#include <random> // mt19937

namespace PLS {
    
    std::vector<std::string> split(const std::string & s, const char separator); {
        size_t i = 0;
        size_t j = s.find(c);
        std::vector<string> result;
        while (j != string::npos) {
            result.push_back(s.substr(i, j-i));
            i = ++j;
            j = s.find(c, j);
        }
        if (j == string::npos) result.push_back(s.substr(i, s.length()));
        return result;
    }


    Mat2D read_matrix_file(const string & filename, const char sep, const bool verbose, std::ostream & os) {
        if (verbose) { os << "Loading " << filename << std::endl; }

        std::ifstream myfile(filename.c_str());
        std::stringstream ss;

        std::vector<std::vector<float_type>> M;
        if (myfile.is_open()) {
            std::string line;

            while (getline(myfile, line) ) {
                //split string based on "," and store results into vector
                std::vector<std::string> fields = split(line, sep);

                std::vector<float_type> row(fields.size());
                for(size_t i = 0; i < fields.size(); i++) { row[i] = stod(fields[i]); }
                if (M.size() != 0) {
                    if (M.back().size() != row.size()) {
                        if (verbose) { os << "Error: row " << M.size() << " has " << row.size() << " columns, but previous row(s) have " << M.back().size() << " columns." << std::endl; }
                        exit(1);
                    }
                }
                M.push_back(row);
            }
        }

        Mat2D X(M.size(), M[0].size());
        for (size_t row = 0; row < M.size(); row++) { X.row(i) = to_evector<Row>(M[i]); }
        return X;
    }

    Row colwise_stdev(const Mat2D & mat, const Row & means ) {
        const float_type N = mat.rows();
        if ( N < 2 ) return Row::Zero(mat.cols());
        // N-1 for unbiased sample variance
        return ((mat.rowwise() - means).array().square().colwise().sum()/(N-1)).sqrt();
    }

    Mat2D colwise_z_scores(const Mat2D & mat, const Row & mean, const Row & stdev) {
        Row local_sd = stdev;
        // sd == 0 => implies all values the same => x_i - mean == 0
        // Technically: z scores are undefined if the stdev is 0 => this should yield nan.
        // However, scores == nan => borks downstream calculations
        // This makes the scores == 0 instead.
        // TODO: change algorithm to not pass this parameter to PLS.
        for (int c = 0; c < local_sd.size(); c++) if (local_sd[c] == 0) local_sd[c] = 1;
        Mat2D mmeans = mat.rowwise() - mean;

        Mat2D zs = mmeans.array().rowwise() / stdev.array();
        return zs;
    };

    Mat2D colwise_z_scores(const Mat2D & mat) {
        const Row means = mat.colwise().mean();
        const Row stdev = colwise_stdev(mat, means);
        return colwise_z_scores(mat, means, stdev);
    };
};

using namespace PLS;

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
float_type dominant_eigenvalue(const EigenSolver<Mat2Dc> es) {
    const size_t idx = find_dominant_ev(es);
    return abs(es.eigenvalues()[idx].real());
};

Colc dominant_eigenvector(const EigenSolver<Mat2D> es) {
    const size_t idx = find_dominant_ev(es);
    return es.eigenvectors().col(idx);
};

//
// Numerical Approximation to Normal Cumulative Distribution Function
//
// DESCRIPTION:
// REFERENCE: Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical
// Tables, U.S. Dept of Commerce - National Bureau of Standards, Editors: M. Abramowitz and I. A. Stegun
// December 1972, p. 932
// INPUT: z=computed Z-value
// OUTPUT: probn=cumulative probability from -infinity to z
//
//
float_type normalcdf(float_type z) {
    const double c1 = 0.196854;
    const double c2 = 0.115194;
    const double c3 = 0.000344;
    const double c4 = 0.019527;
    float_type p;
    if (z < 0) {
        z = -z;
        p = 1 - 0.5 / pow(1 + c1*z + c2*z*z + c3*z*z*z + c4*z*z*z*z, 4);
    } else {
        p = 0.5 / pow(1 + c1*z + c2*z*z + c3*z*z*z + c4*z*z*z*z, 4);
    }
    float_type probn = 1.0 - p;
    return probn;
};

//
// WILCOXON SIGNED RANK TEST FOR EVALUATING RELATIVE QUALITY OF TWO
// COMPETING METHODS
//
// DESCRIPTION: Pairwise comparison between sets of model predictions
// Competing models: model#1, model#2
//
// REFERENCE: Lehmann E. L. Nonparamtrics: Statistical Methods Based on Ranks.
// Holden-Day: San Francisco, 1975, 120-132.
//
// Let: U=sum of postive ranks, V=sum of negative ranks
// (V>U) is evidence that the model#1 is better)
// Define: d=U-V and t=U+V=n(n+1)/2
// Then V=(t-d)/2
//
// Asymptotic Theory: Suppose n is the number of samples.
// Then, E(V)=n(n+1)/4 and Var(V)=n(n+1)(2n+1)/24.
// It follows that (V-E(V))/Std(V) is approx. normally distributed.
//
// INPUT: err_1=prediction errors from model#1
//        err_2=prediction errors from model#2
//
// OUTPUT: probw=Prob{V is larger than observed}
// If probw is small enough, conclude that model#1 is better
//
// Based on Matlab code from
// Thomas E. V. Non-parametric statistical methods for multivariate calibration
// model selection and comparison. J. Chemometrics 2003; 17: 653–659
//
float_type wilcoxon(const Col & err_1, const Col & err_2) {
    assert(err_1.size() == err_2.size());
    size_t n = err_1.rows();
    Col del = err_1.cwiseAbs() - err_2.cwiseAbs();
    Rowi sdel = Rowi::Zero(n);
    for (size_t i = 0; i < n; i++) {
        sdel(i) = (0 < del(i)) - (del(i) < 0); // get the sign of each element
    }
    Col adel = del.cwiseAbs();
    // 's' gives the original positions (indices) of the sorted values
    auto s = ordered(adel);
    float_type d = 0;
    for (size_t i = 0; i < n; i++) { d += static_cast<float_type>(i + 1) * sdel(s[i]); }
    float_type t  = static_cast<float_type>(n * (n + 1)) / 2.0;
    float_type v  = (t - d) / 2.0;
    float_type ev = t/2.0;
    float_type sv = sqrt(static_cast<float_type>(n * (n+1) * (2*n+1)) / 24.0);
    float_type z = (v - ev) / sv;
    float_type probw = 1.0 - normalcdf(z);

    return probw;
};

// creates index vectors for partitioning Eigen::Mat2Ds
// n == sample.size() + complement.size(); k = sample.size()
// full = initially, an iota(0 => n-1); shuffled over and over
// the sample & complement vectors are overwritten with
// new index partition each call
void rand_nchoosek(
    std::mt19937 & rng,
    vector<size_t> & full,
    vector<size_t> & sample,
    vector<size_t> & complement
) {
    std::shuffle(full.begin(), full.end(), rng);
    std::copy(full.begin(), full.begin() + sample.size(), sample.begin());
    std::copy(full.begin() + sample.size(), full.end(), complement().begin());
}


// TODO: several of the loop constructs seem ripe for row/col-wise operations / broadcasting:
// https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html


// "Modified kernel algorithms 1 and 2"
// from Dayal and MacGregor (1997) "Improved PLS Algorithms" J. of Chemometrics. 11,73-85.
PLS_Model& PLS_Model::plsr(const Mat2D& X, const Mat2D& Y, const METHOD algorithm) {
    method = algorithm;
    int M = Y.cols(); // Number of response variables == columns in Y

    if (algorithm == KERNEL_TYPE1) T.setZero(X.rows(), A);

    Mat2D XY = X.transpose() * Y;
    Mat2D XX;
    if (algorithm == KERNEL_TYPE2) XX = X.transpose() * X;

    for (size_t i = 0; i < A; i++) {
        Colc w, p, q, r, t;
        complex<float_type> tt;
        if (M == 1) {
            w = XY.cast<complex<float_type> >();
        } else {
            EigenSolver<Mat2D> es( (XY.transpose() * XY) );
            q = dominant_eigenvector(es);
            w = (XY*q);
        }

        w /= sqrt((w.transpose()*w)(0,0)); // use normalize function from eigen?
        r = w;

        if (i != 0) for (size_t j = 0; j <= i - 1; j++) {
            r -= (P.col(j).transpose()*w)(0,0)*R.col(j);
        }
        
        if (algorithm == KERNEL_TYPE1) {
            t = X*r;
            tt = (t.transpose()*t)(0,0);
            p.noalias() = (X.transpose()*t);
        } else if (algorithm == KERNEL_TYPE2) {
            tt = (r.transpose()*XX*r)(0,0);
            p.noalias() = (r.transpose()*XX).transpose();
        }

        p /= tt;
        q.noalias() = (r.transpose()*XY).transpose(); q /= tt;
        XY -= ((p*q.transpose())*tt).real(); // is casting this to 'real' always safe?
        W.col(i) = w;
        P.col(i) = p;
        Q.col(i) = q;
        R.col(i) = r;
        if (algorithm == KERNEL_TYPE1) T.col(i) = t;
    }



    return *this;
};

const Mat2Dc PLS_Model::scores(const Mat2D& X_new, const size_t comp) const {
    assert (A >= comp);
    return X_new * R.leftCols(comp);
};

const Mat2Dc PLS_Model::coefficients(const size_t comp) const {
    assert (A >= comp);
    return R.leftCols(comp)*Q.leftCols(comp).transpose();
};

const Mat2D PLS_Model::fitted_values(const Mat2D& X, const size_t comp) const {
    assert (A >= comp);
    return X*coefficients(comp).real();
};

const Mat2D PLS_Model::residuals(const Mat2D& X, const Mat2D& Y, const size_t comp) const {
    assert (A >= comp);
    return Y - fitted_values(X, comp);
};

const Row PLS_Model::SSE(const Mat2D& X, const Mat2D& Y, const size_t comp) const {
    return residuals(X, Y, comp).colwise().squaredNorm();
};

// Total sum of squares
const Row PLS_Model::SST(const Mat2D& Y) const {
    Row sst(Y.cols());
    for (size_t c = 0; c < static_cast<size_t>(Y.cols()); c++) {
        sst(c) = (Y.col(c).array() - (Y.col(c).sum()/Y.rows())).square().sum();
    }
    return sst;
}

const Row PLS_Model::explained_variance(const Mat2D& X, const Mat2D& Y, const size_t comp) const {
    assert (A >= comp);
    return (1.0 - SSE(X, Y, comp).cwiseQuotient( SST(Y) ).array()).matrix();
}

// leave-one-out validation of model (i.e., are we overfitting?)
Mat2D PLS_Model::loo_validation(const Mat2D& X, const Mat2D& Y, const VALIDATION_OUTPUT out_type) const {
    Mat2D Xv = X.bottomRows(X.rows()-1);
    Mat2D Yv = Y.bottomRows(Y.rows()-1);

    Mat2D SSEv = Mat2D::Zero(Y.cols(), this->A);

    PLS_Model plsm_v(Xv.cols(), Yv.cols(), this->A);
    for (size_t i = 0; i < static_cast<size_t>(X.rows())-1; i++) {
        // run pls for the data, less one observation
        plsm_v.plsr(Xv, Yv, this->method);
        for (size_t j = 1; j <= this->A; j++) {
            // now see how well the data predict the missing observation
            Row res = plsm_v.residuals(X.row(i), Y.row(i), j).row(0);
            // tally the squared errors
            SSEv.col(j - 1) += res.cwiseProduct(res).transpose();
        }
        Xv.row(i) = X.row(i);
        Yv.row(i) = Y.row(i);
    }
    switch (out_type) {
        case PRESS:
            return SSEv;
        case RMSEP:
            SSEv /= X.rows();
            return SSEv.cwiseSqrt();
        default:
            exit(1);
    }
};

// leave-some-out validation of model (i.e., are we overfitting?)
Mat2D lso_validation(const Mat2D& X, const Mat2D& Y, VALIDATION_OUTPUT out_type, float test_fraction, int num_trials) { 
    const std::vector<Mat2D> Ev = _lso_cv_residual_matrix(X, Y, test_fraction, num_trials);
    assert(Ev.size() > 0);
    const int num_residuals = Ev[0].rows();
    Mat2D SSEv = Mat2D::Zero(Y.cols(), this->A);

    for (int j = 0; j < this->A; j++) {
        Mat2D res = Ev[j];
        // square all of the residuals
        Mat2D SE  = res.cwiseProduct(res);
        // rows in SSEv correspond to different parameters
        // Collapse the squared errors so that we're summing over all predicted rows
        // then transpose, so that rows now represent different parameters
        SSEv.col(j) += SE.colwise().sum().transpose();
    }
    if ( out_type == PRESS ) {
        return SSEv;
    } else {
        SSEv /= num_residuals;
        if ( out_type == MSEP ) { 
            return SSEv;
        } else {
            // RMSEP
            return SSEv.cwiseSqrt();
        }
    }
}
    

std::vector<Mat2D> PLS_Model::_loo_cv_error_matrix(const Mat2D& X, const Mat2D& Y) const {
    Mat2D Xv = X.bottomRows(X.rows()-1);
    Mat2D Yv = Y.bottomRows(Y.rows()-1);

    // vector of error matrices(rows=Y.rows(), cols=Y.cols())
    // col = component #, row = obs #, tier = Y category
    std::vector<Mat2D> Ev(Y.cols(), Mat2D::Zero(X.rows(), this->A));

    PLS_Model plsm_v(Xv.cols(), Yv.cols(), this->A);
    for (size_t row_out = 0; row_out < static_cast<size_t>(X.rows()); row_out++) {
        plsm_v.plsr(Xv, Yv, this->method);
        for (size_t num_comps = 1; num_comps <= this->A; num_comps++) {
            Row res = plsm_v.residuals(X.row(row_out), Y.row(row_out), num_comps).row(0);
            for (int k = 0; k < res.size(); k++) Ev[k](row_out, num_comps-1) = res(k);
        }
        // if not on the last row, swap which row is being left out
        if (row_out < static_cast<size_t>(Xv.rows())) {
            Xv.row(row_out) = X.row(row_out);
            Yv.row(row_out) = Y.row(row_out);
        }
    }
    return Ev;
};

std::vector<Mat2D> PLS_model::_lso_cv_error_matrix(
    const Mat2D & X, const Mat2D & Y, const float_type test_fraction, const size_t num_trials,
    std::mt19937 & rng
) const {
    const size_t N = X.rows();
    const size_t test_size = static_cast<size_t>(test_fraction * N + 0.5);
    const size_t train_size = N - test_size;

    std::vector<Mat2D> Ev(A, Mat2D::Zero(num_trials*test_size, Y.cols()));
    std::vector<size_t> sample(train_size);
    std::vector<size_t> complement(test_size);
    std::vector<size_t> full(sample.size() + complement.size());
    std::iota(full.begin(), full.end(), 0);

    Mat2D Xv(train_size, X.cols()); // values we're training on
    Mat2D Yv(train_size, Y.cols());
    Mat2D Xp(test_size, X.cols());  // values we're predicting
    Mat2D Yp(test_size, Y.cols());

    PLS_Model plsm_v(Xv.cols(), Yv.cols(), this->A);
    for (size_t rep = 0; rep < num_trials; ++rep) {
        rand_nchoosek(rng, full, sample, complement);
        Xv = X(sample, Eigen::placeholders::all);
        Yv = Y(sample, Eigen::placeholders::all);
        Xp = X(complement, Eigen::placeholders::all);
        Yp = Y(complement, Eigen::placeholders::all);
        plsm_v.plsr(Xv, Yv, this->algorithm);
        for (size_t num_comps = 1; num_comps <= this->A; num_comps++) {
            Mat2D res = plsm_v.residuals(Xp, Yp, num_comps);
            Ev[num_comps - 1].middleRows(rep*test_size, test_size) = res; // write to submatrix; middleRows(startRow, numRows)
        }
    }

    return Ev;
};

std::vector<Mat2D> PLS_Model::_new_data_cv_error_matrix(const Mat2D& X_new, const Mat2D& Y_new) const {
    // vector of error matrices(rows=Y.rows(), cols=Y.cols())
    // col = component #, row = obs #, tier = Y category
    std::vector<Mat2D> Ev(Y_new.cols(), Mat2D::Zero(X_new.rows(), this->A));

    for (size_t j = 1; j <= this->A; j++) { // j is component #
        Mat2D res = residuals(X_new, Y_new, j);
        for (size_t k = 0; k < static_cast<size_t>(res.cols()); k++) { // k is Y category
            Ev[k].col(j - 1) = res.col(k);
        }
    }
    return Ev;
};

// if val_method is LOO, X and Y should be original data
// if ... is NEW_DATA, X and Y should be observations not included in the original model
// if ... is LSO, X and Y should be original data, and test_fraction and num_trials
// must be supplied
const Rowsz PLS_Model::optimal_num_components(
    const Mat2D& X, const Mat2D& Y, const VALIDATION_METHOD val_method,
    const float_type test_fraction, const size_t num_trials,
    mt19937 & rng
) const {
    assert((val_method != LSO) or ((num_trials > 0) and (0.0 < test_fraction) and (test_fraction < 1.0)));
    // col = component #, row = obs #, tier = Y category

    std::vector<Mat2D> errors;
    switch(val_method) {
        case LOO: errors = _loo_cv_error_matrix(X, Y); break;
        case NEW_DATA: errors = _new_data_error_matrix(X, Y); break;
        case LSO: errors = _lso_cv_error_matrix(X, Y, test_fraction, num_trials, rng);
        default:
            std::cerr << "Validation method " << val_method << " not supported. " << std::endl;
            exit(-1);
    }

    Mat2D press = Mat2D::Zero(Y.cols(), A);

    // Determine PRESS values
    for (size_t i = 0; i < static_cast<size_t>(errors.size()); i++) {    // for each Y category
        for (size_t k = 0; k < static_cast<size_t>(errors[i].cols()); k++) {  // for each component
            // for each observation
            press(i, k) += errors[i](Eigen::placeholders::all, k).array().square().sum();
        }
    }

    Colsz min_press_idx(press.rows());
    for (size_t r = 0; r < press.rows(); r++) { press.row(r).minCoeff(&min_press_idx[r]); }
    Rowsz best_comp = min_press_idx.array() + 1; // +1 to convert from index to component number
    // Find the min number of components that is not significantly
    // different from the min PRESS at alpha = 0.1 for each Y category
    const float_type ALPHA = 0.1;
    for (size_t i = 0; i < errors.size(); i++) {             // for each Y category
        Col err1 = errors[i].col(min_press_idx(i));          // get errors for the min press # of components
        for (size_t j = 0; j < min_press_idx(i); j++) {      // for fewer # of components
            Col err2 = errors[i].col(j);                     // ... get their errors
            float_type p = wilcoxon(err1, err2);             // determine if error-with-fewer-components is good enough
            if (p > ALPHA) {
                best_comp[i] = j + 1;                        // +1 to convert from index to # of components
                break;
            }
        }
    }
    return best_comp;
};

void PLS_Model::print_explained_variance(const Mat2D& X, const Mat2D& Y, std::ostream& os) const {
    const size_t wd = ceil(std::log10(A));
    for (size_t ncomp = 1; ncomp <= A; ncomp++) {
        // How well did we do with this many components?
        os << std::setw(wd) << ncomp << " components ";
        os << "explained variance: " << explained_variance(X, Y, ncomp);
        //cerr << "root mean squared error of prediction (RMSEP):" << plsm.rmsep(X, Y, A) << endl;
        os << " SSE: " << SSE(X, Y, ncomp) <<  std::endl;
    }
};

void PLS_Model::print_state(std::ostream& os) const {
    //P, W, R, Q, T
    os <<
        "P:"   << std::endl <<
        P << std::endl <<
        "W:"   << std::endl <<
        W << std::endl <<
        "R:"   << std::endl <<
        R << std::endl <<
        "Q:"   << std::endl <<
        Q << std::endl <<
        "T:"   << std::endl <<
        T << std::endl <<
        "coefficients:" << std::endl <<
        coefficients() << std::endl;

};

void PLS_Model::print_model_assessment(
    const Mat2D & X, const Mat2D & Y,
    const size_t training_size, const size_t testing_size,
    const size_t optimal_components, const size_t used_components,
    std::ostream& os
) const {
    os << "Train / Test split: " << training_size << " / " << testing_size << std::endl;
    print_explained_variance(X, Y, os);
    os << "Optimal number of components for each parameter (validation method == NEW DATA):\t" << optimal_components << std::endl;
    os << "Using " << used_components << " components." << std::endl;
};

void PLS_Model::print_validation(
    const Mat2D & X, const Mat2D & Y,
    const size_t training_size, const size_t testing_size,
    std::ostream& os
) {
    os << "Leave-one-out validation:" << std::endl;
    Mat2D looRMSEP = plsm.loo_validation(X, Y, RMSEP);
    os << "RMSE Matrix:" << std::endl << looRMSEP << std::endl;
    os << "Optimal number of components:\t" << plsm.loo_optimal_num_components(X, Y) << std::endl;

    os << "Validation (lso):" << std::endl;
    Mat2D lsoRMSEP = plsm.lso_validation(X, Y, RMSEP, 0.3, 10*X.rows());
    os << lsoRMSEP << std::endl;
    
    os << "Optimal number of components (lso):\t" << plsm.lso_optimal_num_components(X, Y, 0.3, 10*X.rows()) << std::endl;

}    