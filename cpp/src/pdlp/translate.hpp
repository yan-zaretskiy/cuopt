/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/sparse_matrix.hpp>

#include <utilities/copy_helpers.hpp>

#include <limits>
#include <type_traits>
#include <vector>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
static dual_simplex::user_problem_t<i_t, f_t> cuopt_problem_to_simplex_problem(
  raft::handle_t const* handle_ptr, detail::problem_t<i_t, f_t>& model)
{
  dual_simplex::user_problem_t<i_t, f_t> user_problem(handle_ptr);

  int m                  = model.n_constraints;
  int n                  = model.n_variables;
  int nz                 = model.nnz;
  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = cuopt::host_copy(model.objective_coefficients, handle_ptr->get_stream());

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.x = std::vector<f_t>(cuopt::host_copy(model.coefficients, handle_ptr->get_stream()));
  csr_A.j = std::vector<i_t>(cuopt::host_copy(model.variables, handle_ptr->get_stream()));
  csr_A.row_start = std::vector<i_t>(cuopt::host_copy(model.offsets, handle_ptr->get_stream()));

  user_problem.rhs.resize(m);
  user_problem.row_sense.resize(m);
  user_problem.range_rows.clear();
  user_problem.range_value.clear();

  auto model_constraint_lower_bounds =
    cuopt::host_copy(model.constraint_lower_bounds, handle_ptr->get_stream());
  auto model_constraint_upper_bounds =
    cuopt::host_copy(model.constraint_upper_bounds, handle_ptr->get_stream());

  // All constraints have lower and upper bounds
  // lr <= a_i^T x <= ur
  for (int i = 0; i < m; ++i) {
    const double constraint_lower_bound = model_constraint_lower_bounds[i];
    const double constraint_upper_bound = model_constraint_upper_bounds[i];
    if (constraint_lower_bound == constraint_upper_bound) {
      user_problem.row_sense[i] = 'E';
      user_problem.rhs[i]       = constraint_lower_bound;
    } else if (constraint_upper_bound == std::numeric_limits<double>::infinity()) {
      user_problem.row_sense[i] = 'G';
      user_problem.rhs[i]       = constraint_lower_bound;
    } else if (constraint_lower_bound == -std::numeric_limits<double>::infinity()) {
      user_problem.row_sense[i] = 'L';
      user_problem.rhs[i]       = constraint_upper_bound;
    } else {
      // This is range row
      user_problem.row_sense[i] = 'E';
      user_problem.rhs[i]       = constraint_lower_bound;
      user_problem.range_rows.push_back(i);
      const double bound_difference = constraint_upper_bound - constraint_lower_bound;
      user_problem.range_value.push_back(bound_difference);
    }
  }
  user_problem.num_range_rows = user_problem.range_rows.size();
  std::tie(user_problem.lower, user_problem.upper) =
    extract_host_bounds<f_t>(model.variable_bounds, handle_ptr);
  user_problem.problem_name = model.original_problem_ptr->get_problem_name();
  if (model.row_names.size() > 0) {
    user_problem.row_names.resize(m);
    for (int i = 0; i < m; ++i) {
      user_problem.row_names[i] = model.row_names[i];
    }
  }
  if (model.var_names.size() > 0) {
    user_problem.col_names.resize(n);
    for (int j = 0; j < n; ++j) {
      if (j < (int)model.var_names.size()) {
        user_problem.col_names[j] = model.var_names[j];
      } else {
        user_problem.col_names[j] = "_CUOPT_x" + std::to_string(j);
      }
    }
  }
  user_problem.obj_constant = model.presolve_data.objective_offset;
  user_problem.obj_scale    = model.presolve_data.objective_scaling_factor;
  user_problem.var_types.resize(n);

  auto model_variable_types = cuopt::host_copy(model.variable_types, handle_ptr->get_stream());
  for (int j = 0; j < n; ++j) {
    user_problem.var_types[j] =
      model_variable_types[j] == var_t::CONTINUOUS
        ? cuopt::linear_programming::dual_simplex::variable_type_t::CONTINUOUS
        : cuopt::linear_programming::dual_simplex::variable_type_t::INTEGER;
  }

  user_problem.Q_offsets = model.Q_offsets;
  user_problem.Q_indices = model.Q_indices;
  user_problem.Q_values  = model.Q_values;

  if (model.original_problem_ptr->has_quadratic_constraints()) {
    const auto& qcs = model.original_problem_ptr->get_quadratic_constraints();
    cuopt_expects(!qcs.empty(),
                  error_type_t::ValidationError,
                  "Quadratic-constraint flag is set, but no constraints were provided");

    // Use a practical tolerance for text-parsed MPS numeric values.
    const f_t tol = std::numeric_limits<f_t>::epsilon() * 2;

    // SOC conversion accepts only diagonal Lorentz-form QCMATRIX rows:
    //   -x_head^2 + sum_i x_tail_i^2 <= 0.
    // The barrier consumes SOCs as trailing variable blocks [head, tails...], so we validate all
    // QCMATRIX blocks first, then apply a single column permutation to the linear model.
    std::vector<std::vector<i_t>> cone_vars;
    std::vector<i_t> cone_dims;
    std::vector<char> is_cone_var(static_cast<size_t>(n), 0);
    cone_vars.reserve(qcs.size());
    cone_dims.reserve(qcs.size());

    for (const auto& qc : qcs) {
      cuopt_expects(qc.constraint_row_type == 'L',
                    error_type_t::ValidationError,
                    "Only <= quadratic constraints are supported for SOC conversion");
      cuopt_expects(qc.linear_values.empty(),
                    error_type_t::ValidationError,
                    "SOC conversion currently requires zero linear terms in quadratic constraints");
      cuopt_expects(qc.rhs_value < tol && qc.rhs_value > -tol,
                    error_type_t::ValidationError,
                    "SOC conversion currently requires rhs = 0 for quadratic constraints");

      cuopt_expects(qc.quadratic_offsets.size() >= 2,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' has invalid CSR offsets (need at least 2 entries)",
                    qc.constraint_row_name.c_str());
      cuopt_expects(qc.quadratic_values.size() == qc.quadratic_indices.size(),
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' quadratic_values and quadratic_indices length "
                    "mismatch for CSR Q",
                    qc.constraint_row_name.c_str());

      const i_t q_n = static_cast<i_t>(qc.quadratic_values.size());
      cuopt_expects(q_n >= 2,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' SOC must have at least 2 diagonal entries in Q (nnz "
                    "%d)",
                    qc.constraint_row_name.c_str(),
                    static_cast<int>(q_n));

      cuopt_expects(
        qc.quadratic_offsets.size() == static_cast<size_t>(n) + 1,
        error_type_t::ValidationError,
        "Quadratic constraint '%s' Q must be n by n in CSR: expected %zu CSR row pointers (offsets "
        "length n+1), got %zu (n = %d)",
        qc.constraint_row_name.c_str(),
        static_cast<size_t>(n) + 1,
        qc.quadratic_offsets.size(),
        static_cast<int>(n));
      cuopt_expects(
        qc.quadratic_offsets[static_cast<size_t>(n)] == q_n,
        error_type_t::ValidationError,
        "Quadratic constraint '%s' Q last CSR offset %d must equal number of nonzeros (nnz) %d for "
        "this diagonal Q",
        qc.constraint_row_name.c_str(),
        static_cast<int>(qc.quadratic_offsets[static_cast<size_t>(n)]),
        static_cast<int>(q_n));
      cuopt_expects(qc.quadratic_offsets[0] == 0,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' Q CSR offsets[0] must be 0",
                    qc.constraint_row_name.c_str());

      // Verify Q: n by n CSR, diagonal entries only, Lorentz pattern.
      // Scan each row r: empty or one nnz on (r,r) with value -1 (head) or +1 (tail);
      // tail order follows this scan; no requirement that diagonal indices be sorted.
      i_t head     = static_cast<i_t>(-1);
      i_t n_head_m = 0;
      std::vector<i_t> tail_row_vars{};
      tail_row_vars.reserve(static_cast<size_t>(q_n - 1));

      for (i_t r = 0; r < n; ++r) {
        const i_t p_beg = qc.quadratic_offsets[static_cast<size_t>(r)];
        const i_t p_end = qc.quadratic_offsets[static_cast<size_t>(r + 1)];
        cuopt_expects(p_beg >= 0 && p_beg <= p_end && p_end <= q_n,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' Q row %d has invalid CSR offsets [%d, %d)",
                      qc.constraint_row_name.c_str(),
                      static_cast<int>(r),
                      static_cast<int>(p_beg),
                      static_cast<int>(p_end));

        if (p_beg == p_end) { continue; }

        cuopt_expects(p_beg + 1 == p_end,
                      error_type_t::ValidationError,
                      "Quadratic constraint '%s' Q row %d: expected at most one stored entry on "
                      "the diagonal per "
                      "row (got end - beg = %d)",
                      qc.constraint_row_name.c_str(),
                      static_cast<int>(r),
                      static_cast<int>(p_end - p_beg));

        const i_t col = qc.quadratic_indices[static_cast<size_t>(p_beg)];
        const f_t v   = qc.quadratic_values[static_cast<size_t>(p_beg)];
        cuopt_expects(
          col == r,
          error_type_t::ValidationError,
          "Quadratic constraint '%s' Q row %d: only main diagonal (j,j) entries are allowed; got "
          "column %d",
          qc.constraint_row_name.c_str(),
          static_cast<int>(r),
          static_cast<int>(col));

        const f_t neg_one_delta = v + f_t(1);
        const f_t pos_one_delta = v - f_t(1);
        const bool is_neg_one   = (neg_one_delta >= -tol && neg_one_delta <= tol);
        const bool is_pos_one   = (pos_one_delta >= -tol && pos_one_delta <= tol);
        if (is_neg_one) {
          ++n_head_m;
          head = r;
        } else if (is_pos_one) {
          tail_row_vars.push_back(r);
        } else {
          cuopt_expects(false,
                        error_type_t::ValidationError,
                        "Quadratic constraint '%s' Q row %d: diagonal for SOC must be -1 (head) or "
                        "+1 (tail); got "
                        "%.17g",
                        qc.constraint_row_name.c_str(),
                        static_cast<int>(r),
                        static_cast<double>(v));
        }
      }
      cuopt_expects(
        n_head_m == 1,
        error_type_t::ValidationError,
        "Quadratic constraint '%s' SOC Q: expected exactly one diagonal with value -1 (cone head), "
        "found %d",
        qc.constraint_row_name.c_str(),
        static_cast<int>(n_head_m));
      cuopt_expects(
        static_cast<i_t>(tail_row_vars.size()) == q_n - 1,
        error_type_t::ValidationError,
        "Quadratic constraint '%s' SOC Q: expected %d diagonals with value +1 (tails), found %zu",
        qc.constraint_row_name.c_str(),
        static_cast<int>(q_n - 1),
        tail_row_vars.size());
      cuopt_expects(head >= 0,
                    error_type_t::ValidationError,
                    "Quadratic constraint '%s' SOC Q: internal error (head index invalid)",
                    qc.constraint_row_name.c_str());

      std::vector<i_t> cone;
      cone.reserve(static_cast<size_t>(q_n));
      cone.push_back(head);
      cone.insert(cone.end(), tail_row_vars.begin(), tail_row_vars.end());
      for (const i_t var : cone) {
        cuopt_expects(!is_cone_var[static_cast<size_t>(var)],
                      error_type_t::ValidationError,
                      "Variable %d appears in more than one SOC QCMATRIX block; overlapping cones "
                      "are not supported",
                      static_cast<int>(var));
        is_cone_var[static_cast<size_t>(var)] = 1;
      }
      cone_dims.push_back(q_n);
      cone_vars.push_back(std::move(cone));
    }

    std::vector<i_t> old_to_new(static_cast<size_t>(n), i_t{-1});
    std::vector<i_t> new_to_old;
    new_to_old.reserve(static_cast<size_t>(n));
    for (i_t j = 0; j < n; ++j) {
      if (is_cone_var[static_cast<size_t>(j)]) { continue; }
      old_to_new[static_cast<size_t>(j)] = static_cast<i_t>(new_to_old.size());
      new_to_old.push_back(j);
    }
    const i_t cone_var_start = static_cast<i_t>(new_to_old.size());
    for (const auto& cone : cone_vars) {
      for (const i_t old_j : cone) {
        old_to_new[static_cast<size_t>(old_j)] = static_cast<i_t>(new_to_old.size());
        new_to_old.push_back(old_j);
      }
    }
    cuopt_expects(static_cast<i_t>(new_to_old.size()) == n,
                  error_type_t::RuntimeError,
                  "Internal error while building SOC variable permutation");

    for (i_t row = 0; row < csr_A.m; ++row) {
      for (i_t p = csr_A.row_start[static_cast<size_t>(row)];
           p < csr_A.row_start[static_cast<size_t>(row + 1)];
           ++p) {
        const i_t old_j = csr_A.j[static_cast<size_t>(p)];
        cuopt_expects(old_j >= 0 && old_j < n,
                      error_type_t::ValidationError,
                      "Linear constraint matrix column index %d is outside [0, %d)",
                      static_cast<int>(old_j),
                      static_cast<int>(n));
        csr_A.j[static_cast<size_t>(p)] = old_to_new[static_cast<size_t>(old_j)];
      }
    }

    auto permute_dense_by_old_to_new = [&](auto& values, const char* name) {
      if (values.empty()) { return; }
      using value_t = typename std::decay_t<decltype(values)>::value_type;
      cuopt_expects(values.size() == static_cast<size_t>(n),
                    error_type_t::ValidationError,
                    "%s length %zu does not match number of variables %d",
                    name,
                    values.size(),
                    static_cast<int>(n));
      std::vector<value_t> permuted(values.size());
      for (i_t old_j = 0; old_j < n; ++old_j) {
        permuted[static_cast<size_t>(old_to_new[static_cast<size_t>(old_j)])] =
          std::move(values[static_cast<size_t>(old_j)]);
      }
      values = std::move(permuted);
    };

    permute_dense_by_old_to_new(user_problem.objective, "objective");
    permute_dense_by_old_to_new(user_problem.lower, "lower bounds");
    permute_dense_by_old_to_new(user_problem.upper, "upper bounds");
    permute_dense_by_old_to_new(user_problem.var_types, "variable types");
    permute_dense_by_old_to_new(user_problem.col_names, "column names");

    if (!user_problem.Q_values.empty()) {
      cuopt_expects(user_problem.Q_indices.size() == user_problem.Q_values.size(),
                    error_type_t::ValidationError,
                    "Quadratic objective indices and values length mismatch");
      cuopt_expects(user_problem.Q_offsets.size() == static_cast<size_t>(n) + 1,
                    error_type_t::ValidationError,
                    "Quadratic objective CSR offsets length must be n+1 when SOC QCMATRIX "
                    "conversion permutes variables");
      cuopt_expects(user_problem.Q_offsets[0] == 0,
                    error_type_t::ValidationError,
                    "Quadratic objective CSR offsets[0] must be 0");
      cuopt_expects(user_problem.Q_offsets[static_cast<size_t>(n)] ==
                      static_cast<i_t>(user_problem.Q_values.size()),
                    error_type_t::ValidationError,
                    "Quadratic objective CSR last offset must equal number of nonzeros");

      std::vector<i_t> q_offsets(static_cast<size_t>(n) + 1, 0);
      for (i_t old_row = 0; old_row < n; ++old_row) {
        const i_t p_beg = user_problem.Q_offsets[static_cast<size_t>(old_row)];
        const i_t p_end = user_problem.Q_offsets[static_cast<size_t>(old_row + 1)];
        cuopt_expects(
          p_beg >= 0 && p_beg <= p_end && p_end <= static_cast<i_t>(user_problem.Q_values.size()),
          error_type_t::ValidationError,
          "Quadratic objective CSR offsets are invalid at row %d",
          static_cast<int>(old_row));
        const i_t new_row                           = old_to_new[static_cast<size_t>(old_row)];
        q_offsets[static_cast<size_t>(new_row + 1)] = p_end - p_beg;
      }
      for (i_t row = 0; row < n; ++row) {
        q_offsets[static_cast<size_t>(row + 1)] += q_offsets[static_cast<size_t>(row)];
      }

      std::vector<i_t> q_indices(user_problem.Q_indices.size());
      std::vector<f_t> q_values(user_problem.Q_values.size());
      auto q_write = q_offsets;
      for (i_t old_row = 0; old_row < n; ++old_row) {
        const i_t new_row = old_to_new[static_cast<size_t>(old_row)];
        for (i_t p = user_problem.Q_offsets[static_cast<size_t>(old_row)];
             p < user_problem.Q_offsets[static_cast<size_t>(old_row + 1)];
             ++p) {
          const i_t old_col = user_problem.Q_indices[static_cast<size_t>(p)];
          cuopt_expects(old_col >= 0 && old_col < n,
                        error_type_t::ValidationError,
                        "Quadratic objective column index %d is outside [0, %d)",
                        static_cast<int>(old_col),
                        static_cast<int>(n));
          const i_t dst                       = q_write[static_cast<size_t>(new_row)]++;
          q_indices[static_cast<size_t>(dst)] = old_to_new[static_cast<size_t>(old_col)];
          q_values[static_cast<size_t>(dst)]  = user_problem.Q_values[static_cast<size_t>(p)];
        }
      }

      user_problem.Q_offsets = std::move(q_offsets);
      user_problem.Q_indices = std::move(q_indices);
      user_problem.Q_values  = std::move(q_values);
    }

    user_problem.cone_var_start         = cone_var_start;
    user_problem.second_order_cone_dims = std::move(cone_dims);
  }

  csr_A.to_compressed_col(user_problem.A);

  return user_problem;
}

template <typename i_t, typename f_t>
void translate_to_crossover_problem(const detail::problem_t<i_t, f_t>& problem,
                                    optimization_problem_solution_t<i_t, f_t>& sol,
                                    dual_simplex::lp_problem_t<i_t, f_t>& lp,
                                    dual_simplex::lp_solution_t<i_t, f_t>& initial_solution)
{
  CUOPT_LOG_DEBUG("Starting translation");

  auto stream                     = problem.handle_ptr->get_stream();
  std::vector<f_t> pdlp_objective = cuopt::host_copy(problem.objective_coefficients, stream);

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(
    problem.n_constraints, problem.n_variables, problem.nnz);
  csr_A.x         = std::vector<f_t>(cuopt::host_copy(problem.coefficients, stream));
  csr_A.j         = std::vector<i_t>(cuopt::host_copy(problem.variables, stream));
  csr_A.row_start = std::vector<i_t>(cuopt::host_copy(problem.offsets, stream));

  stream.synchronize();
  CUOPT_LOG_DEBUG("Converting to compressed column");
  csr_A.to_compressed_col(lp.A);
  CUOPT_LOG_DEBUG("Converted to compressed column");

  std::vector<f_t> slack(problem.n_constraints);
  std::vector<f_t> tmp_x = cuopt::host_copy(sol.get_primal_solution(), stream);
  stream.synchronize();
  dual_simplex::matrix_vector_multiply(lp.A, f_t(1.0), tmp_x, f_t(0.0), slack);
  CUOPT_LOG_DEBUG("Multiplied A and x");

  lp.A.col_start.resize(problem.n_variables + problem.n_constraints + 1);
  lp.A.x.resize(problem.nnz + problem.n_constraints);
  lp.A.i.resize(problem.nnz + problem.n_constraints);
  i_t nz = problem.nnz;
  for (i_t j = problem.n_variables; j < problem.n_variables + problem.n_constraints; ++j) {
    lp.A.col_start[j] = nz;
    lp.A.i[nz]        = j - problem.n_variables;
    lp.A.x[nz]        = -1.0;
    ++nz;
  }
  lp.A.col_start[problem.n_variables + problem.n_constraints] = nz;
  CUOPT_LOG_DEBUG("Finished with A");

  const i_t n = problem.n_variables + problem.n_constraints;
  const i_t m = problem.n_constraints;
  lp.num_cols = n;
  lp.num_rows = m;
  lp.A.n      = n;
  lp.rhs.resize(m, 0.0);
  lp.lower.resize(n);
  lp.upper.resize(n);
  lp.obj_constant = problem.presolve_data.objective_offset;
  lp.obj_scale    = problem.presolve_data.objective_scaling_factor;

  auto [lower, upper] = extract_host_bounds<f_t>(problem.variable_bounds, problem.handle_ptr);

  std::vector<f_t> constraint_lower = cuopt::host_copy(problem.constraint_lower_bounds, stream);
  std::vector<f_t> constraint_upper = cuopt::host_copy(problem.constraint_upper_bounds, stream);

  lp.objective.resize(n, 0.0);
  std::copy(
    pdlp_objective.begin(), pdlp_objective.begin() + problem.n_variables, lp.objective.begin());
  std::copy(lower.begin(), lower.begin() + problem.n_variables, lp.lower.begin());
  std::copy(upper.begin(), upper.begin() + problem.n_variables, lp.upper.begin());

  problem.handle_ptr->get_stream().synchronize();
  for (i_t i = 0; i < m; ++i) {
    lp.lower[problem.n_variables + i] = constraint_lower[i];
    lp.upper[problem.n_variables + i] = constraint_upper[i];
  }
  CUOPT_LOG_DEBUG("Finished with lp");

  initial_solution.resize(m, n);

  std::copy(tmp_x.begin(), tmp_x.begin() + problem.n_variables, initial_solution.x.begin());
  for (i_t j = problem.n_variables; j < n; ++j) {
    initial_solution.x[j] = slack[j - problem.n_variables];
    // Project slack variables inside their bounds
    if (initial_solution.x[j] < lp.lower[j]) { initial_solution.x[j] = lp.lower[j]; }
    if (initial_solution.x[j] > lp.upper[j]) { initial_solution.x[j] = lp.upper[j]; }
  }
  CUOPT_LOG_DEBUG("Finished with x");
  initial_solution.y = cuopt::host_copy(sol.get_dual_solution(), stream);

  std::vector<f_t> tmp_z = cuopt::host_copy(sol.get_reduced_cost(), stream);
  stream.synchronize();
  std::copy(tmp_z.begin(), tmp_z.begin() + problem.n_variables, initial_solution.z.begin());
  for (i_t j = problem.n_variables; j < n; ++j) {
    initial_solution.z[j] = initial_solution.y[j - problem.n_variables];
  }
  CUOPT_LOG_DEBUG("Finished with z");

  CUOPT_LOG_DEBUG("Finished translating");
}

}  // namespace cuopt::linear_programming
