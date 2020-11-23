import argparse
import csv
import itertools as it
import numpy as np
from scipy.optimize import least_squares
import sys
import textwrap
from typing import Callable


def mat_to_eqs(x: list, mat: np.array) -> Callable:
    """\
    Create equations system using information on pairwise correlation
    coefficients encoded in symmetric matrix.

    Each equation has the format:
      h_a * h_b - pairwise_cc = 0
    with:
      h_a, h_b: unknown variables (1-dimensional vectors).
      pairwise_cc: known pairwise correlation coefficient.

    The resulting equations system is overdetermined, if there are more
    equations than unknown variables.

    :param x: list of N unknown variables [h_0, h_1, h_2, ..., h_N-1]
              representing the hypothetical values for each dataset
              whose products h_a * h_b approximate the given pairwise
              correlation coefficients.
              (needed for generating equations system,
               NOT to be specified by user)
    :param mat: square matrix.
                upper off-diagonal elements encode all given pairwise
                correlation coefficients.
                (the rest of the square matrix is ignored.)
    :return: list of equations.
    """
    # Initialise list of equations.
    result = []

    # Loop over elements above main diagonal of given square matrix:
    for row_idx, row in enumerate(mat[:-1]):
        h_a_idx = row_idx
        for column_idx, cell in enumerate(row[row_idx + 1 :], row_idx + 1):
            h_b_idx = column_idx
            pairwise_cc = cell
            # If cell element is not NaN, then create new equation and
            # add it to equations system.
            if np.isfinite(pairwise_cc):
                # Append new equation to memory.
                result.append(x[h_a_idx] * x[h_b_idx] - pairwise_cc)

    return result


def orig_to_eqs(x: np.array,
                h_a_idx: int,
                h: np.array,
                orig_mat_row: np.array) -> Callable:
    """\
    Create equations system using information on originally known
    pairwise correlation coefficients (parsed from infile) and the
    representative result vectors.

    Each equation has the format:
      h_a * h_b - pairwise_cc = 0
    with:
      h_a: unknown variable
           (n-dimensional column-vector)
      h_b: known representative vector
           (n-dimensional column-vector)
      pairwise_cc: known pairwise correlation coefficient

    The resulting equations system is overdetermined, if there are more
    equations than unknown elements.

    :param x: unknown n-dimensional column-vector
              (needed for generating equations system,
               NOT to be specified by user).
    :param h_a_idx: index of currently optimised representative result
                    vector.
    :param h: matrix with row-wise listing of representative result
              vectors.
    :param orig_mat_row: matrix-row of originally parsed pairwise
                         correlation coefficients.
    """
    # Initialise list of equations.
    result = []

    # Get h_a that belongs to the current orig_mat_row/h_a_idx.
    h_a = h[h_a_idx]

    # For all pairwise_cc in the given orig_mat_row.
    for column_idx, cell in enumerate(orig_mat_row):
        h_b_idx = column_idx
        pairwise_cc = cell
        # If cell element is not NaN, then create new equation and add
        # it to equations system.
        if np.isfinite(pairwise_cc):
            # Get h_b for the current not-NaN cell of the original
            # correction matrix.
            h_b = h[h_b_idx]
            # Append new equation to memory.
            result.append(sum(x * h_b) - pairwise_cc)

    return result


def initialise_distrusts(hyp: list, mat: np.array) -> list:
    """\
    Initialise a distrust-score for each h_i and store them in a list.

    The distrust-score is determined for each h_i by processing its
    assigned row:
      distrust = con_hyp_num - pro_hyp_num
    with:
      con_hyp_num: number of non-NaN cells that contradict hypothesis
                   (in the current row)
      pro_hyp_num: number of non-NaN cells that support hypothesis
                   (in the current row)

    :param hyp: list of signs for all h_i in the encoding:
                  *  1: positive
                  *  0: zero
                  * -1: negative
    :param mat: symmetric matrix.
                off-diagonal elements encode all given pairwise
                correlation coefficients.
    :return: list of row-wise distrust-scores for all h_i.
    """
    # Initialise list of distrust-scores.
    result = []

    # Loop over symmetric matrix:
    for h_a_idx, row in enumerate(mat):
        h_a_sign = hyp[h_a_idx]
        # Initialise counters for current row.
        con_hyp_num = 0
        pro_hyp_num = 0
        # Update counters for the current row.
        for h_b_idx, cell in enumerate(row):
            h_b_sign = hyp[h_b_idx]
            # Only process non-NaN cells.
            if np.isfinite(cell):
                cell_sign = np.sign(cell)
                # Sign of cell matches hypothesis.
                if cell_sign == h_a_sign * h_b_sign:
                    pro_hyp_num += 1
                # Sign of cell does NOT match hypothesis.
                else:
                    con_hyp_num += 1

        # Calculate distrust-score for current row.
        distrust = con_hyp_num - pro_hyp_num
        # Append distrust-score to memory.
        result.append(distrust)

    return result


def optimise_hypothesis(hyp: list, distrusts: list, mat: np.array) -> list:
    """\
    Optimise hypothesis concerning the sign of the hypothetical value
    h_i of each dataset by interpreting the given pairwise correlation
    coefficients as scalar products.

    :param hyp: list of signs for all h_i in the encoding:
                  *  1: positive
                  *  0: zero
                  * -1: negative
    :param distrusts: list of row-wise distrust-scores for all h_i:
                        distrust = con_hyp_num - pro_hyp_num
                      with:
                        con_hyp_num: number of non-NaN cells that
                                     contradict hypothesis
                                     (in the current row)
                        pro_hyp_num: number of non-NaN cells that support
                                     hypothesis.
                                     (in the current row)
    :param mat: symmetric matrix.
                off-diagonal elements encode all given pairwise
                correlation coefficients.
    :return: list of optimised signs for all h_i.
             explains the maximum number of signs of the given pairwise
             correlation coefficients.
    """
    # Get index and value of maximum distrust-score.
    # Index is equivalent to according row-index
    max_distrust_idx, max_distrust = max(enumerate(distrusts),
                                         key=lambda p: p[1])

    # If hypothesis is not optimal yet:
    if max_distrust > 0:
        # Toggle sign for h_i with maximum distrust-score.
        hyp[max_distrust_idx] *= -1
        # Update distrust-score for h_i whose sign was toggled by
        # inverting its sign,
        # because the toggled sign causes all previously incorrect signs
        # to become correct and vice versa.
        distrusts[max_distrust_idx] *= -1
        # Also update distrust-scores for all other h_i whose signs
        # were NOT changed.
        # The distrust-score can only be changed by the column that is
        # assigned to the one h_i whose sign was toggled.
        h_a_sign = hyp[max_distrust_idx]
        for notmax_distrust_idx in it.chain(range(max_distrust_idx),
                                            range(max_distrust_idx + 1,
                                                  len(distrusts))):
            h_b_sign = hyp[notmax_distrust_idx]
            cell = mat[max_distrust_idx, notmax_distrust_idx]
            # Distrust-score is only changed, if the cell value is
            # not NaN.
            if np.isfinite(cell):
                cell_sign = np.sign(cell)
                # Sign of cell matches hypothesis.
                if cell_sign == h_a_sign * h_b_sign:
                    # Distrust is decreased by 2, because there is
                    # 1 more value supporting hypothesis and 1 less
                    # value contradicting hypothesis.
                    distrusts[notmax_distrust_idx] -= 2
                # Sign of cell does NOT match hypothesis.
                else:
                    # Distrust is increased by 2, because there is
                    # 1 less value supporting hypothesis and 1 more
                    # value contradicting hypothesis.
                    distrusts[notmax_distrust_idx] += 2

        # Further optimisation necessary.
        return optimise_hypothesis(hyp, distrusts, mat)
        
    # If hypothesis is optimal already, then return list of signs.
    else:
        return hyp


def root_mean_square(mat_a: np.array, mat_b: np.array) -> float:
    """\
    Calculate root-mean-square for the two given matrices.
    (Order of matrices does NOT matter.)

    :param mat_a: matrix A.
    :param mat_b: matrix B.
    :return: root-mean-square.
    """
    # Calculate squared-deviates for all matrix-elements.
    # Order of matrices does not matter because of square.
    # Resulting matrix may contain NaN-value(s).
    squared_deviates_mat = (mat_b - mat_a) ** 2
    # Calculate root-mean-square.
    # NaN-values are ignored.
    rms = np.sqrt(np.nanmean(squared_deviates_mat))
    return rms

    
# ----------------------------------------------------------------------
# Parse command-line arguments.

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description=textwrap.dedent("""\
    Represent M-dimensional data as n-dimensional data (n < M) for each of
    the N datasets.  This results in N representative n-dimensional vectors.

    The M-dimensional data of the N datasets is input/encoded by a NxN-
    matrix of pairwise correlation coefficients.  This matrix is used to
    calculate a representative n-dimensional vector for each of the N
    datasets.
    """))
parser.add_argument(
    "dim", type=int,
    help=textwrap.dedent("""\
    integer n
    for dimension of representative result vectors
    """))
parser.add_argument(
    "infile", type=str,
    help=textwrap.dedent("""\
    input-file
    encoding the matrix of pairwise correlation coefficients
    with each line in the space-separated format:
      '[dataset_a_num] [dataset_b_num] [pairwise_cc]'
    with:
      dataset_a_num: number of dataset a
      dataset_b_num: number of dataset b
      pairwise_cc:   pairwise correlation coefficient between
                     datasets a and b.

    'N * (N-1) / 2' pairwise correlation coefficients resp.
    unique combinations of N datasets are possible.  However,
    this program is able to tolerate the absence of some
    pairwise correlation coefficients at the expense of
    precision by estimating their values.
    """))
parser.add_argument(
    "-lsm", action='store_true',
    help=textwrap.dedent("""\
    additionally calculate and output (STDERR) intermediate
    results with alternative method using the least-squares
    approach (for comparison with the used iterative approach)

    The intermediate results are needed for filling in the
    off-diagonal elements and any other missing elements of
    the correlation matrix.  These values are the 1-dimensional
    representations of the datasets, when the correlation
    matrix is interpreted as a matrix of products.

    The least-squares method increases the runtime, especially
    for large N.  It will cause a memory-error, when N is too
    large.  The intermediate results of the least-squares
    methods are NOT used for the subsequent computation.
    """))
args = parser.parse_args()

# ----------------------------------------------------------------------
# Get information on NxN-matrix of pairwise correlation coefficients by
# parsing infile.  Store parsed information in intermediate storage.
# Encoded matrix does NOT contain main diagonal elements and it is
# possible that some off-diagonal elements are missing.

# Intermediate storage for information of parsed infile.
# List of tuples, each tuple in the format:
# ([dataset_a_num], [dataset_b_num], [pairwise_cc])
parsed_info = []

# Parse infile of the space-separated format:
# '[dataset_a_num] [dataset_b_num] [pairwise_cc]'
with open(args.infile, newline='') as infile:
#    infile_parsed = csv.reader(infile_raw, delimiter=' ')
    # Convert values to correct types and store as tuples.
    for line in infile:
        dataset_a_num, dataset_b_num, pairwise_cc = line.split()
        parsed_info.append((int(dataset_a_num),
                            int(dataset_b_num),
                            float(pairwise_cc)))

# ----------------------------------------------------------------------
# Use parsed information of infile to create symmetric NxN-matrix.

# Get width N of square matrix by identifing the maximum dataset number.
mat_width = max([max(dataset_a_num, dataset_b_num)
                 for dataset_a_num, dataset_b_num, _ in parsed_info])

# Initialise empty square matrix of correct size by making use of
# calculated width.
mat = np.full((mat_width, mat_width), np.nan)

# Fill square matrix symetrically with correlation coefficients.
# The finished matrix contains all the information of the intermediate
# storage pairwise_cc, but has a more convenient structure and it stores
# its elements in an ordered fashion.
for dataset_a_num, dataset_b_num, cc in parsed_info:
    dataset_a_idx = dataset_a_num - 1
    dataset_b_idx = dataset_b_num - 1
    mat[dataset_a_idx, dataset_b_idx] = cc
    mat[dataset_b_idx, dataset_a_idx] = cc

# Delete intermediate storage, as it is not needed anymore.
parsed_info.clear

# FB.
print(textwrap.dedent("""\
      ===
      Correlation matrix parsed from infile:
      {}""").format(mat), file=sys.stderr, flush=True)

# Make a copy in order to memorise the original state of the parsed
# matrix without the future estimated values, that will fill in the
# now empty (=NaN) elements.
orig_mat = mat.copy()

# ----------------------------------------------------------------------
# Calculate correction factor for 2nd and higher eigenvalue(s).
# This correction is NOT needed for the 1st eigenvalue, because the
# unknown (=NaN) values of the matrix are approximated by presuming
# 1-dimensional vectors as the basis of the matrix interpretation as dot
# products.

# Calculate total number of matrix elements.
mat_elements_total = mat_width ** 2

# Count the number of unknown (=NaN) elements.
mat_elements_unknown = np.count_nonzero(np.isnan(mat))
# Calculate the number of known (= not-NaN) elements.
mat_elements_known = mat_elements_total - mat_elements_unknown

# Calculate correction factor.
correction_factor = mat_elements_known / mat_elements_total

# FB.
print(textwrap.dedent("""\
      ===
      Correction factor for 2nd and higher eigenvalue(s):
      {0:3.4f}""").format(correction_factor), file=sys.stderr, flush=True)

# ----------------------------------------------------------------------
# Calculate hypothetical value (1-dimensional vector) h_i for each
# dataset by interpreting the given correlation coefficients as scalar
# products.

# FB.
print(textwrap.dedent("""\
      ===
      Interpretation of correlation matrix as dot product matrix:"""),
      file=sys.stderr, flush=True)

# ---
# Least-squares-method:
# Most precise method, but is slow (might even raise memory-error) for
# large N.
# For comparison only, since its results will NOT be used for the
# following computation.
if args.lsm:
    # For this only one half of the symmetric matrix is needed, as the
    # other half is identical.
    # The resulting equations system can be solved for N >= 3 and is
    # overdetermined for N >= 4.
    # Find the least-squares-solution for this (overdetermined)
    #  equationssystem (equations in the form: h_a * h_b = cc).
    # For each h_i: use arbitrary value of 0.5 as initial guess for
    #               least_squares-method.
    h_values_lsm = least_squares(mat_to_eqs,
                                 [0.5] * mat_width,
                                 args=(mat,)).x
    # FB.
    print(textwrap.dedent("""\
          ---
          all h_i by least-squares method (not used, only for comparison):
          {}""").format(np.around(h_values_lsm, decimals=4)),
          file=sys.stderr, flush=True)

# ---
# Quicker alternative method using iterations.

# Memory for current hypothesis concerning sign of each h_i.
# List of signs for all h_i in the encoding:
#   *  1: positive
#   *  0: zero
#   * -1: negative
# Initial hypothesis: all signs are positive.
h_signs = [1] * mat_width

# Estimate signs for each h_i by refining hypothesis on signs.
h_signs = optimise_hypothesis(h_signs, initialise_distrusts(h_signs, mat), mat)

# Estimate absolute values for each h_i by determining sqrt of mean of
# non-NaN absolute values for every row.
h_abs = np.sqrt(np.nanmean(np.absolute(mat), axis=1))

# Combine estimated signs with absolute values in obtain total value for
# each h_i.
h_values = h_signs * h_abs
h_values = h_values.reshape((1, mat_width))

# FB.
print(textwrap.dedent("""\
      ---
      all h_i by iterative approach:
      initial values:
      {}""").format(np.around(h_values[0], decimals=4)),
      file=sys.stderr, flush=True)

# ----------------------------------------------------------------------
# Complement symmetric matrix by using the scalar products of estimated
# values of h_i to replace NaN-cells.

# Matrix positions that have estimated values
# (only for diagonal and upper off-diagonal values, due to the symmetry
#  the positions of the lower-diagonal values can be inferred).
# List of tuples (row_idx, column_idx).
estimated_positions = []

# For off-diagonal cells.
for row_idx in range(mat_width - 1):
    for column_idx in range(row_idx + 1, mat_width):
        cell = mat[row_idx, column_idx]
        if np.isnan(cell):
            # Calculate scalar product as new cell value.
            cell = h_values[0, row_idx] * h_values[0, column_idx]
            # Fill in new cell value in actual cell and its symmetric
            # partner cell.
            mat[row_idx, column_idx] = cell
            mat[column_idx, row_idx] = cell
            # Memorise positions of estimated values.
            estimated_positions.append((row_idx, column_idx))
# For diagonal cells.
for diagonal_idx in range(mat_width):
    # Calculate scalar product as new cell value.
    cell = h_values[0, diagonal_idx] ** 2
    # Fill in new cell value.
    mat[diagonal_idx, diagonal_idx] = cell
    # Memorise position of estimated value.
    estimated_positions.append((diagonal_idx, diagonal_idx))

# Refine total values of each h_i:
# Initialise h_values of the hypothetical non-existant previous iteration
# with the correct format but with impossible values.
# Needed for exit condition of otherwise endless loop.
h_values_old = np.array([[np.nan] * mat_width])
# Counter for how many refinement steps were done.
iteration_num = 0
# FB.
print("refinement by iteration:", file=sys.stderr, flush=True)
# Repeat refinement until the values of all h_i do not significantly
# change anymore.
while True:
    for h_idx in range(mat_width):
        new_h = np.sum(h_values * mat[h_idx]) / np.sum(h_values ** 2)
        h_values[0, h_idx] = new_h
    # FB.
    print("#{}: {}".format(iteration_num,
                                np.around(h_values[0], decimals=4)),
          end="\r", file=sys.stderr, flush=True)
    # Update values of estimated positions.
    for row_idx, column_idx in estimated_positions:
        new_val = h_values[0, row_idx] * h_values[0, column_idx]
        mat[column_idx, row_idx] = new_val
        mat[row_idx, column_idx] = new_val

    iteration_num += 1

    # Exit endless loop as soon as new values for all h_i are similar to
    # those of the last iteration.
    if np.allclose(h_values[0], h_values_old[0], rtol=0, atol=1e-05):
        print("", file=sys.stderr, flush=True)  # FB.
        break

    # Memorise h_values for comparison in the next iteration.
    h_values_old = np.copy(h_values)
    
# ----------------------------------------------------------------------
# Use complemented symmetric matrix to calculate final representative
# vectors.

# Eigendecomposition.
eig_vals, eig_vecs = np.linalg.eigh(mat, UPLO='U')

# Sort eigenvalue-eigenvector-pairs in non-ascending order based on the
# eigenvalue.
eig_pairs = list(zip(eig_vals, eig_vecs.T))
eig_pairs = sorted(eig_pairs, key=lambda p: p[0], reverse=True)

# FB.
print(textwrap.dedent("""\
      ===
      Uncorrected eigenvalue(s):
      {} used:
      {}
      {} unused:
      {}""").format(args.dim,
                    np.around([eig_val for eig_val, eig_vec
                               in eig_pairs[ : args.dim]],
                              decimals=4),
                    len(eig_pairs) - args.dim,
                    np.around([eig_val for eig_val, eig_vec
                               in eig_pairs[args.dim : ]],
                              decimals=4)),
      file=sys.stderr, flush=True)

# Initialise empty matrix in the expected size Nxn for storing final
# representative vectors.
# (Each matrix-row is a representative vector.)
rep_mat = np.full((mat_width, args.dim), np.nan)

# FB.
corrected_eig_vals = np.full_like(eig_vals[: args.dim], np.nan)

# Fill matrix of final representative vectors.
for eig_pair_idx, eig_pair in enumerate(eig_pairs[: args.dim]):
    eig_val, eig_vec = eig_pair
    eig_pair_num = eig_pair_idx + 1
    # For the 2nd and higher eigenvalue(s).
    if eig_pair_num >= 2:
        # Correct eigenvalue with correction factor.
        eig_val /= correction_factor
    # FB.
    corrected_eig_vals[eig_pair_idx] = eig_val
    # Calculate and fill in representative vector.
    rep_mat[:, eig_pair_idx] = np.sqrt(eig_val) * eig_vec
# FB.
print(textwrap.dedent("""\
---
Corrected eigenvalue(s):
{} used:
{}""").format(args.dim,
              np.around(corrected_eig_vals, decimals=4)),
      file=sys.stderr, flush=True)

# ----------------------------------------------------------------------
# Report on root-mean-square BEFORE the following refinement of
# representative vectors.

# Dot product of representative vectors yields a matrix with values
# approximating the correlation matrix.
dot_mat = rep_mat.dot(rep_mat.T)
# Calculate root-mean-square between approximation and correlation
# matrix.
rms = root_mean_square(dot_mat, orig_mat)

# FB.
print('{0:>4} {1:>8} {2:>8} {3:>8}'.
      format("iter", "RMS", "max_chg", "rms_chg"),
      file=sys.stderr, flush=True)
print('{0:>4} {1:>8.5f} {2:>8} {3:>8}'.
      format("0", rms, "-", "-"),
      file=sys.stderr, flush=True)

# ----------------------------------------------------------------------
# Refine representative vectors by minimising sum-of-squared-deviates
# between dot product matrix obtained from representative vectors and
# the original correlation matrix, that was parsed from the pairwise
# correlation coefficients of the infile.

# Maximum number of refinement-iterations is arbitrarily set to 20.
for iter_idx in range(1, 21):
    # Remember values before the refinement of the current iteration.
    rep_mat_old = np.copy(rep_mat)
    dot_mat_old = np.copy(dot_mat)

    # For all rows/h_a of the original matrix of correlation coefficients.
    for row_idx, row in enumerate(orig_mat):
        h_a_idx = row_idx
        h_a = rep_mat[h_a_idx]
        # Find least-squares-solution for difference between current row
        # of originally input pairwise correlation coefficients and the
        # dot product of the according representative vectors.
        h_a_lsm = least_squares(orig_to_eqs,
                                h_a,
                                args=(h_a_idx, rep_mat, row)).x
        # Update h_a with optimised value.
        rep_mat[h_a_idx] = h_a_lsm

    # ------------------------------------------------------------------
    # Report on root-mean-square AFTER the refinement of the current
    # iteration.

    # Dot product of representative vectors yields a matrix with values
    # approximating the correlation matrix.
    dot_mat = rep_mat.dot(rep_mat.T)
    # Calculate root-mean-square between approximation and correlation
    # matrix.
    rms = root_mean_square(dot_mat, orig_mat)

    # Calculate maximum change of representative vector elements of
    # current iteration.
    chg_mat = rep_mat - rep_mat_old
    max_chg_flatidx = np.argmax(np.absolute(chg_mat))
    max_chg_idx = np.unravel_index(max_chg_flatidx, chg_mat.shape)
    max_chg = chg_mat[max_chg_idx]

    # Calculate root-mean-square between current and previous estimation.
    rms_chg = root_mean_square(dot_mat, dot_mat_old)

    # FB.
    print('{0:>4} {1:>8.5f} {2:>8.5f} {3:>8.5f}'.
          format(iter_idx, rms, max_chg, rms_chg),
          file=sys.stderr, flush=True)

    # If the current refinement-iteration does not noticeably change
    # the element values of the representative vectors.
    # The representative vectors will be rounded to only 4 decimals in
    # the final output, so the comparison here works with a precision of
    # 6 decimals in order to be safe.
    if np.allclose(max_chg, 0, atol=1e-06):
        # Revert values of representative vectors to those prior to
        # current refinement-iteration.
        rep_mat = np.copy(rep_mat_old)
        # Stop refinement process.
        break

# ----------------------------------------------------------------------
# Write final representative vectors from matrix to STDOUT.

for row_num, row in enumerate(rep_mat, 1):
    rep_num = row_num
    rep_vec = row
    print("{0:>6}".format(row_num), end='')
    for rep_vec_element in rep_vec:
        print(" {0:7.4f}".format(rep_vec_element), end='')
    print()

# Save representative vectors
np.save('rep_mat.npy', rep_mat)
# FB.
print(textwrap.dedent("""\
      ===
      Success! Finished outputting {}-dimensional representative vectors! :D
      """).format(args.dim),
      file=sys.stderr, flush=True)




# --------------------------------------------------------------------
# DOWN BELOW IS CODE WE HAVE ADDED
# Save rep_mat to results.txt
import numpy as np
np.savetxt('results.txt', rep_mat, delimiter='\t')
