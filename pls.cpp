#include <fstream> // ifstream
#include <cmath> // log10, ceil
#include <iomanip> // setw
#include <random> // mt19937
#include "pls.h"

using namespace PLS;

namespace PLS {
    
    std::vector<std::string> split(const std::string & s, const char separator) {
        size_t i = 0;
        size_t j = s.find(separator);
        std::vector<std::string> result;
        while (j != std::string::npos) {
            result.push_back(s.substr(i, j-i));
            i = ++j;
            j = s.find(separator, j);
        }
        if (j == std::string::npos) result.push_back(s.substr(i, s.length()));
        return result;
    }


    Mat2D read_matrix_file(
        const std::string & filename, const char separator
    ) {

        std::ifstream myfile(filename);
        std::stringstream ss;

        std::vector<std::vector<float_type>> M;
        if (myfile.is_open()) {
            std::string line;

            while (getline(myfile, line) ) {
                //split string based on "," and store results into vector
                std::vector<std::string> fields = split(line, separator);

                std::vector<float_type> row(fields.size());
                for(size_t i = 0; i < fields.size(); i++) { row[i] = stod(fields[i]); }
                if (M.size() != 0) {
                    if (M[0].size() != row.size()) {
                        std::cerr << "Error: row " << M.size() << " has " << row.size() << " columns, but previous row(s) have " << M[0].size() << " columns." << std::endl;
                        exit(1);
                    }
                }
                M.push_back(row);
            }
        }

        Mat2D X(M.size(), M[0].size());
        for (size_t row = 0; row < M.size(); row++) { X.row(row) = to_evector<Row>(M[row]); }
        return X;
    }

    Row SST(const Mat2D& mat, const Row & means) {
        const float_type N = mat.rows();
        if ( N < 2 ) return Row::Zero(mat.cols());
        return ((mat.rowwise() - means)).array().square().colwise().sum();
    }

    Row SST(const Mat2D & mat) {
        return SST(mat, mat.colwise().mean());
    }

    Row colwise_stdev(const Mat2D & mat, const Row & means) {
        const float_type N = mat.rows();
        // N-1 for unbiased sample variance
        return (SST(mat, means)/(N-1)).array().sqrt();
    }

    Row colwise_stdev(const Mat2D & mat) {
        return colwise_stdev(mat, mat.colwise().mean());
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

    template<typename MATTYPE>
    size_t find_dominant_ev(const EigenSolver<MATTYPE> & es) {
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

    // extract the dominant eigenvalue from EigenSolver
    float_type dominant_eigenvalue(const EigenSolver<Mat2Dc> & es) {
        const size_t idx = find_dominant_ev(es);
        return abs(es.eigenvalues()[idx].real());
    };

    // extract the dominant eigenvector from EigenSolver
    Colc dominant_eigenvector(const EigenSolver<Mat2D> & es) {
        const size_t idx = find_dominant_ev(es);
        return es.eigenvectors().col(idx);
    };


    // Numerical Approximation to Normal Cumulative Distribution Function
    //
    // DESCRIPTION:
    // REFERENCE: Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical
    // Tables, U.S. Dept of Commerce - National Bureau of Standards, Editors: M. Abramowitz and I. A. Stegun
    // December 1972, p. 932
    // INPUT: z = computed Z-value
    // OUTPUT: cumulative probability from -infinity to z (i.e. P(X <= z))
    float_type normalcdf(const float_type z) {
        const double c1 = 0.196854;
        const double c2 = 0.115194;
        const double c3 = 0.000344;
        const double c4 = 0.019527;
        const float_type zstar = abs(z);
        float_type p = 0.5 / pow(1 + c1*zstar + c2*zstar*zstar + c3*zstar*zstar*zstar + c4*zstar*zstar*zstar*zstar, 4);
        return z < 0 ? p : 1.0 - p;
    };

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
        std::vector<Eigen::Index> & full,
        std::vector<Eigen::Index> & sample,
        std::vector<Eigen::Index> & complement
    ) {
        std::shuffle(full.begin(), full.end(), rng);
        std::copy(full.begin(), full.begin() + sample.size(), sample.begin());
        std::copy(full.begin() + sample.size(), full.end(), complement.begin());
    }

    // validation of model based on errors (i.e., are we overfitting?)
    // 
    // errors: Y-components (vector indices), observations (Mat2D rows), num of components (Mat2D cols)
    // with coefficients = residual
    // out_type: how to summarize the errors
    // return: a matrix, rows = Y-components, cols = num of components, coefficients = summarized error
    Mat2D validation(
        const PLSError & errors,
        const PLS::VALIDATION_OUTPUT out_type
    ) {
        if (errors.size() == 0) { return Mat2D::Zero(0, 0); }
        // sum-of-squared error, for each Y component (rows), for 1-to-A components (cols)
        Mat2D SSEv = Mat2D::Zero(errors.size(), errors[0].cols());

        for (size_t y_comp = 0; y_comp < errors.size(); y_comp++) {
            Mat2D res = errors[y_comp];
            // square all of the residuals - cols = components, rows = observations
            Mat2D SE  = res.cwiseProduct(res);
            // rows in SSEv correspond to different "observations" in the cross-validation
            // Collapse the squared errors so that we're summing over all observations
            // then transpose, so that rows now represent different parameters
            SSEv.row(y_comp) += SE.colwise().sum().transpose();
        }

        const size_t num_cv_observations = errors[0].rows();
        switch (out_type) {
            case RESS: return SSEv;
            case MSE: SSEv /= num_cv_observations; return SSEv;
            default: SSEv /= num_cv_observations; return SSEv.cwiseSqrt();
        }

    }

    // return: a row, columns correspondings to columns in Y (used in error(X, Y, ...))
    // each row_i = the optimal number of components to impute a row_i
    Colsz optimal_num_components(
        const PLSError & errors,
        const float_type ALPHA
    ) {
        // rows = Y component, cols = # of components
        Mat2D press = validation(errors, PLS::RESS);

        Colsz min_press_idx(press.rows());

        // Find the min number of components that is not significantly
        // different from the min PRESS at alpha = 0.1 for each Y category
        for (size_t y_comp = 0; y_comp < errors.size(); y_comp++) {   // for each Y category
            press.row(y_comp).minCoeff(&min_press_idx[y_comp]);       // ...find the absolute min PRESS (index)     
            const size_t ref_min = min_press_idx[y_comp];             // ...using that as a reference
            Col err1 = errors[y_comp].col(ref_min);                   // ...get the reference errors
            for (size_t alt_min = 0; alt_min < ref_min; alt_min++) {  // ...for each fewer # of components
                Col err2 = errors[y_comp].col(alt_min);               // ...get their errors
                if (wilcoxon(err1, err2) > ALPHA) {                   // ...if error-with-fewer-components is not significantly different
                    min_press_idx[y_comp] = alt_min; break;           // ...use that instead
                }
            }
        }
        return min_press_idx + Colsz::Constant(press.rows(), 1); // convert index (from 0) to component #s (from 1)
    };

    void print_validation(
        const PLSError & errors,
        const PLS::VALIDATION_METHOD method,
        const PLS::VALIDATION_OUTPUT out_type,
        std::ostream& os
    ) {
        switch (method) {
            case PLS::LOO: os << "Leave-one-out "; break;
            case PLS::LSO: os << "Leave-some-out "; break;
            case PLS::NEW_DATA: os << "New Data "; break;
            default: os << "UNKNOWN ";
        };
        os << "Validation:" << std::endl;
        Mat2D em = validation(errors, out_type);
        switch (out_type) {
            case PLS::MSE: os << "RMSE "; em = em.cwiseSqrt(); break;
            case PLS::RESS: os << "PRESS "; break;
            default: os << "UNKNOWN ";
        }
        os << " Matrix:" << std::endl << em << std::endl;
        os << "Optimal number of components:\t" << optimal_num_components(errors) << std::endl;
    };

};

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
    return X*coefficients(comp).real();
};

const Mat2D PLS_Model::residuals(const Mat2D& X, const Mat2D& Y, const size_t comp) const {
    return Y - fitted_values(X, comp);
};

const Row PLS_Model::SSE(const Mat2D& X, const Mat2D& Y, const size_t comp) const {
    return residuals(X, Y, comp).colwise().squaredNorm();
};

const Row PLS_Model::explained_variance(
    const Mat2D& X, const Mat2D& Y, const size_t comp
) const {
    return (
        1.0 - (SSE(X, Y, comp).array() / SST(Y).array())
    ).matrix(); // 1 - SSE/SST, using eigen broadcasting
}

// template specialization for LOO error
template <>
PLSError PLS_Model::error<PLS::LOO>(
    const Mat2D & X, const Mat2D & Y
) const {
    Mat2D Xv = X.bottomRows(X.rows()-1);
    Mat2D Yv = Y.bottomRows(Y.rows()-1);

    // vector of error matrices(rows=Y.rows(), cols=Y.cols())
    // col = component #, row = obs #, tier = Y category
    PLSError Ev(Y.cols(), Mat2D::Zero(X.rows(), this->A));

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

// template specialization for NEW_DATA error
template <>
PLSError PLS_Model::error<PLS::NEW_DATA>(
    const Mat2D & X, const Mat2D & Y
) const {
    // vector of error matrices(rows=Y.rows(), cols=Y.cols())
    // col = component #, row = obs #, tier = Y category
    PLSError Ev(Y.cols(), Mat2D::Zero(X.rows(), this->A));

    for (size_t num_comps = 1; num_comps <= this->A; num_comps++) { // j is component #
        Mat2D res = residuals(X, Y, num_comps);
        // if Ev were (A, N, Y), would be Ev[num_comps-1] = res;
        for (size_t ycomp = 0; ycomp < static_cast<size_t>(res.cols()); ycomp++) { // k is Y category
            Ev[ycomp].col(num_comps - 1) = res.col(ycomp);
        }
    }
    return Ev;
};

// template specialization for LSO error
template <>
PLSError PLS_Model::error<PLS::LSO>(
    const Mat2D & X, const Mat2D & Y,
    const float_type test_fraction, const size_t num_trials, std::mt19937 & rng
) const {
    const size_t N = X.rows();
    const size_t test_size = static_cast<size_t>(test_fraction * N + 0.5);
    const size_t train_size = N - test_size;

    PLSError Ev(Y.cols(), Mat2D::Zero(num_trials*test_size, A));

    std::vector<Eigen::Index> sample(train_size);
    std::vector<Eigen::Index> complement(test_size);
    std::vector<Eigen::Index> full(sample.size() + complement.size());
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
        plsm_v.plsr(Xv, Yv, this->method);
        for (size_t num_comps = 1; num_comps <= this->A; num_comps++) {
            Mat2D res = plsm_v.residuals(Xp, Yp, num_comps);
            for (size_t ycomp = 0; ycomp < static_cast<size_t>(res.cols()); ycomp++) {
                Ev[ycomp].middleRows(rep*test_size, test_size).col(num_comps - 1) += res.col(ycomp);
            }
        }
    }

    return Ev;
};

// template generalization for not LOO/NEW_DATA error methods - just fail
// TODO figure out how to make this a compile error / warning
template <PLS::VALIDATION_METHOD val_method>
PLSError PLS_Model::error(
    const Mat2D& X, const Mat2D& Y
) const {
    std::cerr << "error<" << val_method <<"> must be provided additional arguments." << std::endl;
    exit(-1);
}

// template generalization for not LSO error methods - just fail
// TODO figure out how to make this a compile error / warning
template <PLS::VALIDATION_METHOD val_method>
PLSError PLS_Model::error(
    const Mat2D& X, const Mat2D& Y,
    const float_type test_fraction, const size_t num_trials, std::mt19937 & rng
) const {
    std::cerr << "error<" << val_method <<"> provided too many arguments." << std::endl;
    exit(-1);
}


void PLS_Model::print_explained_variance(
    const Mat2D& X, const Mat2D& Y, std::ostream& os
) const {
    const size_t wd = ceil(std::log10(A));
    for (size_t ncomp = 1; ncomp <= A; ncomp++) {
        // How well did we do with this many components?
        os << std::setw(wd) << ncomp << " components explained variance: ";
        os << explained_variance(X, Y, ncomp);
        //cerr << "root mean squared error of prediction (RMSEP):" << plsm.rmsep(X, Y, A) << endl;
        os << "  - SSE: " << SSE(X, Y, ncomp) <<  std::endl;
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