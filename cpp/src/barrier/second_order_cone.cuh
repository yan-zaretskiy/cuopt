/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/copy_helpers.hpp>
#include <utilities/cuda_helpers.cuh>

#include <rmm/device_uvector.hpp>

#include <raft/core/device_span.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cub/device/device_segmented_reduce.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// ---------------------------------------------------------------------------
// Flat cone kernels: segmented reductions compute per-cone scalars, then a
// single elementwise launch applies the result across all packed cone entries.
// This keeps the cone math vectorized instead of one block per cone.
// ---------------------------------------------------------------------------

constexpr int flat_block_dim = 256;

template <typename i_t, typename f_t>
__global__ void apply_hinv2_write_kernel(raft::device_span<const f_t> v,
                                         raft::device_span<f_t> out,
                                         raft::device_span<const f_t> w_bar,
                                         raft::device_span<const f_t> inv_eta,
                                         raft::device_span<const f_t> tail_dot,
                                         raft::device_span<const i_t> cone_offsets,
                                         raft::device_span<const i_t> element_cone_ids,
                                         f_t output_scale)
{
  i_t flat_idx = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (flat_idx >= static_cast<i_t>(out.size())) return;

  i_t cone      = element_cone_ids[flat_idx];
  i_t cone_off  = cone_offsets[cone];
  i_t local_idx = flat_idx - cone_off;

  f_t ie_sq     = inv_eta[cone] * inv_eta[cone];
  f_t u_tv      = w_bar[cone_off] * v[cone_off] - tail_dot[cone];
  f_t coeff     = f_t(2) * u_tv * ie_sq;
  int sign      = (local_idx == 0) * 2 - 1;
  f_t value     = coeff * w_bar[flat_idx] - ie_sq * v[flat_idx];
  out[flat_idx] = output_scale * value * sign;
}

template <typename f_t>
struct corrector_raw_t {
  f_t zeta;
  f_t xi;
  f_t psi;
};

template <typename f_t>
struct corrector_raw_sum_t {
  HD corrector_raw_t<f_t> operator()(const corrector_raw_t<f_t>& lhs,
                                     const corrector_raw_t<f_t>& rhs) const
  {
    return {lhs.zeta + rhs.zeta, lhs.xi + rhs.xi, lhs.psi + rhs.psi};
  }
};

template <typename i_t, typename f_t>
struct cone_scratch_t {
  i_t K;
  rmm::device_uvector<corrector_raw_t<f_t>> corrector_raw;  // [K] {zeta, xi, psi}
  rmm::device_uvector<f_t> scalar_slots;  // [6 * K] reusable K-length scalar scratch slots
  rmm::device_uvector<f_t> step_alpha_primal;
  rmm::device_uvector<f_t> step_alpha_dual;
  rmm::device_uvector<std::uint8_t> segmented_reduce_workspace;

  cone_scratch_t(i_t K_in, rmm::cuda_stream_view stream)
    : K(K_in),
      corrector_raw(K_in, stream),
      scalar_slots(6 * K_in, stream),
      step_alpha_primal(K_in, stream),
      step_alpha_dual(K_in, stream),
      segmented_reduce_workspace(0, stream)
  {
  }

  raft::device_span<f_t> hinv2_tail_dot() { return slot_span(0); }
  raft::device_span<f_t> step_s_du1_sq() { return slot_span(0); }
  raft::device_span<f_t> step_s_u1du1() { return slot_span(1); }
  raft::device_span<f_t> step_s_u1_sq() { return slot_span(2); }
  raft::device_span<f_t> step_l_du1_sq() { return slot_span(3); }
  raft::device_span<f_t> step_l_u1du1() { return slot_span(4); }
  raft::device_span<f_t> step_l_u1_sq() { return slot_span(5); }

  raft::device_span<f_t> nt_s1_sq() { return slot_span(0); }
  raft::device_span<f_t> nt_l1_sq() { return slot_span(1); }
  raft::device_span<f_t> nt_sl() { return slot_span(2); }

  raft::device_span<f_t> step_alpha_primal_span() { return cuopt::make_span(step_alpha_primal); }
  raft::device_span<f_t> step_alpha_dual_span() { return cuopt::make_span(step_alpha_dual); }

 private:
  raft::device_span<f_t> slot_span(i_t slot)
  {
    return raft::device_span<f_t>(scalar_slots.data() + slot * K, K);
  }
};

template <typename i_t, typename f_t>
__global__ void fused_corrector_write_kernel(raft::device_span<const f_t> s,
                                             raft::device_span<const f_t> lambda,
                                             raft::device_span<const f_t> dx_aff,
                                             raft::device_span<const f_t> omega,
                                             raft::device_span<const f_t> w_bar,
                                             raft::device_span<const f_t> inv_eta,
                                             raft::device_span<const f_t> inv_1pw0,
                                             raft::device_span<const f_t> rho,
                                             raft::device_span<const corrector_raw_t<f_t>> raw,
                                             raft::device_span<f_t> out,
                                             raft::device_span<const i_t> cone_offsets,
                                             raft::device_span<const i_t> element_cone_ids,
                                             f_t sigma_mu,
                                             f_t output_scale)
{
  i_t flat_idx = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (flat_idx >= static_cast<i_t>(out.size())) return;

  i_t cone         = element_cone_ids[flat_idx];
  i_t cone_off     = cone_offsets[cone];
  i_t local_idx    = flat_idx - cone_off;
  f_t ie           = inv_eta[cone];
  f_t ipw          = inv_1pw0[cone];
  f_t w0           = w_bar[cone_off];
  f_t omega0       = omega[cone_off];
  f_t dx_a0        = dx_aff[cone_off];
  auto raw_vals    = raw[cone];
  f_t coeff_a      = -dx_a0 + raw_vals.zeta * ipw;
  f_t dx0          = (w0 * dx_a0 - raw_vals.zeta) * ie;
  f_t dz0          = -omega0 - dx0;
  f_t w_sq_sum     = max(f_t(0), w0 * w0 - f_t(1));
  f_t w_omega_sum  = f_t(0.5) * (ie * s[cone_off] - lambda[cone_off] / ie);
  f_t omega_sq_sum = max(f_t(0), omega0 * omega0 - rho[cone]);
  f_t omega_dx_sum = ie * (raw_vals.xi + coeff_a * w_omega_sum);
  f_t dx_sq_sum =
    ie * ie * (raw_vals.psi + f_t(2) * coeff_a * raw_vals.zeta + coeff_a * coeff_a * w_sq_sum);
  f_t r_K_0 = (omega0 * omega0 + omega_sq_sum) + (dx0 * dz0 - omega_dx_sum - dx_sq_sum) - sigma_mu;
  f_t nu    = (f_t(2) * omega0 - dx0) * omega_sq_sum - (omega0 + f_t(2) * dx0) * omega_dx_sum;
  f_t inv_rho    = f_t(1) / rho[cone];
  f_t corr0      = (omega0 * r_K_0 - nu) * inv_rho;
  f_t inv_omega0 = f_t(1) / omega0;
  f_t c_inv      = (nu * inv_omega0 - r_K_0) * inv_rho;
  f_t p1         = c_inv + f_t(2) - dx0 * inv_omega0;
  f_t p2         = -(f_t(1) + f_t(2) * dx0 * inv_omega0);
  f_t w_dx_sum   = ie * (raw_vals.zeta + coeff_a * w_sq_sum);
  f_t zeta2      = p1 * w_omega_sum + p2 * w_dx_sum;
  f_t coeff_c    = -corr0 + zeta2 * ipw;

  if (local_idx == 0) {
    out[flat_idx] = output_scale * ((w0 * corr0 - zeta2) * ie);
    return;
  }

  f_t dx_j      = (dx_aff[flat_idx] + coeff_a * w_bar[flat_idx]) * ie;
  f_t corr_j    = p1 * omega[flat_idx] + p2 * dx_j;
  out[flat_idx] = output_scale * ((corr_j + coeff_c * w_bar[flat_idx]) * ie);
}

// ---------------------------------------------------------------------------
// Flattened NT scaling / step-length kernels.
// All follow the same pattern: segmented reduction to per-cone scalars, then
// flat or scalar kernels to write the packed cone outputs.
// ---------------------------------------------------------------------------

template <typename i_t, typename f_t>
__global__ void nt_scaling_scalar_kernel(raft::device_span<const f_t> s,
                                         raft::device_span<const f_t> lambda,
                                         raft::device_span<const i_t> cone_offsets,
                                         raft::device_span<const f_t> s1_sq,
                                         raft::device_span<const f_t> l1_sq,
                                         raft::device_span<const f_t> sl,
                                         raft::device_span<f_t> inv_eta,
                                         raft::device_span<f_t> inv_1pw0,
                                         raft::device_span<f_t> w_bar,
                                         raft::device_span<f_t> omega,
                                         raft::device_span<f_t> rho,
                                         i_t K)
{
  i_t cone = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cone >= K) return;

  i_t off = cone_offsets[cone];
  f_t s0  = s[off];
  f_t l0  = lambda[off];

  f_t s_J       = sqrt(max(f_t(0), s0 * s0 - s1_sq[cone]));
  f_t l_J       = sqrt(max(f_t(0), l0 * l0 - l1_sq[cone]));
  f_t inv_s_J   = f_t(1) / s_J;
  f_t inv_l_J   = f_t(1) / l_J;
  f_t rho_val   = s_J * l_J;
  f_t inv_eta_v = sqrt(l_J / s_J);
  f_t scale     = sqrt(rho_val);

  f_t s_dot_l = (s0 * l0 + sl[cone]) * inv_s_J * inv_l_J;
  f_t gamma   = sqrt(max(f_t(0), (f_t(1) + s_dot_l) * f_t(0.5)));
  f_t inv_2g  = f_t(1) / (f_t(2) * gamma);
  f_t sb0     = s0 * inv_s_J;
  f_t lb0     = l0 * inv_l_J;

  f_t w0         = (sb0 + lb0) * inv_2g;
  inv_eta[cone]  = inv_eta_v;
  inv_1pw0[cone] = f_t(1) / (f_t(1) + w0);
  w_bar[off]     = w0;
  omega[off]     = gamma * scale;
  rho[cone]      = rho_val;
}

template <typename i_t, typename f_t>
__global__ void nt_scaling_tail_kernel(raft::device_span<const f_t> s,
                                       raft::device_span<const f_t> lambda,
                                       raft::device_span<const f_t> inv_eta,
                                       raft::device_span<const f_t> rho,
                                       raft::device_span<f_t> w_bar,
                                       raft::device_span<f_t> omega,
                                       raft::device_span<const i_t> cone_offsets,
                                       raft::device_span<const i_t> element_cone_ids)
{
  i_t flat_idx = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (flat_idx >= static_cast<i_t>(w_bar.size())) return;

  i_t cone     = element_cone_ids[flat_idx];
  i_t cone_off = cone_offsets[cone];
  if (flat_idx == cone_off) return;

  f_t s0          = s[cone_off];
  f_t l0          = lambda[cone_off];
  f_t inv_eta_val = inv_eta[cone];
  f_t rho_val     = rho[cone];
  f_t scale       = sqrt(rho_val);

  f_t s_J     = scale / inv_eta_val;
  f_t l_J     = scale * inv_eta_val;
  f_t inv_s_J = f_t(1) / s_J;
  f_t inv_l_J = f_t(1) / l_J;

  f_t gamma  = omega[cone_off] / scale;
  f_t inv_2g = f_t(1) / (f_t(2) * gamma);
  f_t sb0    = s0 * inv_s_J;
  f_t lb0    = l0 * inv_l_J;
  f_t D      = sb0 + lb0 + f_t(2) * gamma;
  f_t inv_D  = f_t(1) / D;
  f_t c_s    = (gamma + sb0) * inv_D;
  f_t c_l    = (gamma + lb0) * inv_D;

  f_t w_from_s           = inv_2g * inv_s_J;
  f_t w_from_lambda      = -inv_2g * inv_l_J;
  f_t omega_s_coeff      = scale * c_l * inv_s_J;
  f_t omega_lambda_coeff = scale * c_s * inv_l_J;

  f_t sj          = s[flat_idx];
  f_t lj          = lambda[flat_idx];
  w_bar[flat_idx] = w_from_s * sj + w_from_lambda * lj;
  omega[flat_idx] = omega_s_coeff * sj + omega_lambda_coeff * lj;
}

template <typename f_t>
DI f_t cone_step_length_from_scalars(f_t u0, f_t du0, f_t du1_sq, f_t u1du1, f_t c, f_t alpha_max)
{
  f_t a     = du0 * du0 - du1_sq;
  f_t b     = u0 * du0 - u1du1;
  f_t disc  = b * b - a * c;
  f_t alpha = alpha_max;

  if (du0 < f_t(0)) { alpha = min(alpha, -u0 / du0); }
  if ((a > f_t(0) && b > f_t(0)) || disc < f_t(0)) {
    return alpha;
  } else if (a == f_t(0)) {
    if (b < f_t(0)) { alpha = min(alpha, c / (f_t(-2) * b)); }
  } else if (c == f_t(0)) {
    alpha = (a >= f_t(0)) ? alpha : f_t(0);
  } else {
    f_t t  = -(b + copysign(sqrt(disc), b));
    f_t r1 = c / t;
    f_t r2 = t / a;
    if (r1 < f_t(0)) { r1 = alpha; }
    if (r2 < f_t(0)) { r2 = alpha; }
    alpha = min(alpha, min(r1, r2));
  }
  return alpha;
}

template <typename i_t, typename f_t>
__global__ void step_length_pair_kernel(raft::device_span<const f_t> s,
                                        raft::device_span<const f_t> ds,
                                        raft::device_span<const f_t> lambda,
                                        raft::device_span<const f_t> dlambda,
                                        raft::device_span<f_t> alpha_primal,
                                        raft::device_span<f_t> alpha_dual,
                                        raft::device_span<const f_t> s_du1_sq,
                                        raft::device_span<const f_t> s_u1du1,
                                        raft::device_span<const f_t> s_u1_sq,
                                        raft::device_span<const f_t> l_du1_sq,
                                        raft::device_span<const f_t> l_u1du1,
                                        raft::device_span<const f_t> l_u1_sq,
                                        raft::device_span<const i_t> cone_offsets,
                                        f_t alpha_max,
                                        i_t K)
{
  i_t cone = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cone >= K) return;

  i_t off = cone_offsets[cone];
  f_t s_c = max(f_t(0), s[off] * s[off] - s_u1_sq[cone]);
  f_t l_c = max(f_t(0), lambda[off] * lambda[off] - l_u1_sq[cone]);

  alpha_primal[cone] = cone_step_length_from_scalars<f_t>(
    s[off], ds[off], s_du1_sq[cone], s_u1du1[cone], s_c, alpha_max);
  alpha_dual[cone] = cone_step_length_from_scalars<f_t>(
    lambda[off], dlambda[off], l_du1_sq[cone], l_u1du1[cone], l_c, alpha_max);
}

/**
 * Device storage for second-order cone topology, NT scaling, and iterate views.
 *
 * Flat arrays are packed by cone: elements [cone_offsets[i], cone_offsets[i+1])
 * belong to cone i, which has dimension cone_dims[i].
 *
 * Primal/dual iterates (s, lambda) are non-owning spans, pre-sliced by the
 * caller to cover the cone portion of the global x/z vectors.  The caller
 * must keep the underlying memory alive.
 *
 * Only persistent cone state lives here. Reusable per-iteration workspace sits
 * under `scratch`, which keeps the mutating temporary buffers out of the
 * persistent NT state.
 */
template <typename i_t, typename f_t>
struct cone_data_t {
  // --- Topology (set once at construction) ---
  i_t K;    // number of second-order cones
  i_t m_c;  // total cone dimension = sum of cone_dims

  rmm::device_uvector<i_t> cone_offsets;   // [K+1] prefix sums of cone_dims
  rmm::device_uvector<i_t> cone_dims;      // [K]   dimension q_i of each cone
  rmm::device_uvector<i_t> block_offsets;  // [K+1] prefix sums of q_i^2 (for dense block build)
  rmm::device_uvector<i_t> block_entry_cone_ids;  // [sum q_i^2] owning cone id for each block entry

  // --- Primal/dual cone iterates (non-owning views, set by caller) ---
  raft::device_span<f_t> s;       // [m_c] cone slack: s_i in int(Q^{q_i})
  raft::device_span<f_t> lambda;  // [m_c] cone dual:  lambda_i in int(Q^{q_i})

  // --- NT scaling state (recomputed each iteration from s, lambda) ---
  rmm::device_uvector<f_t>
    inv_eta;  // [K]   1/eta_i where eta_i = (||s_i||_J / ||lambda_i||_J)^{1/2}
  rmm::device_uvector<f_t> inv_1pw0;  // [K]   cached 1/(1 + wbar_0_i)
  rmm::device_uvector<f_t> w_bar;     // [m_c] NT scaling direction, unit J-norm, packed by cone
  rmm::device_uvector<f_t> omega;  // [m_c] scaled variable omega_i = H_i^{-1} s_i, packed by cone
  rmm::device_uvector<f_t> rho;    // [K]   ||omega_i||^2_J = ||s_i||_J * ||lambda_i||_J
  rmm::device_uvector<i_t> element_cone_ids;  // [m_c] owning cone id for each packed entry
  cone_scratch_t<i_t, f_t> scratch;

  cone_data_t(i_t K_in,
              const std::vector<i_t>& dims,
              raft::device_span<f_t> s_in,
              raft::device_span<f_t> lambda_in,
              rmm::cuda_stream_view stream)
    : K(K_in),
      m_c(std::accumulate(dims.begin(), dims.end(), i_t(0))),
      cone_offsets(K_in + 1, stream),
      cone_dims(K_in, stream),
      block_offsets(K_in + 1, stream),
      block_entry_cone_ids(
        std::accumulate(
          dims.begin(), dims.end(), i_t(0), [](i_t acc, i_t q) { return acc + q * q; }),
        stream),
      s(s_in),
      lambda(lambda_in),
      inv_eta(K_in, stream),
      inv_1pw0(K_in, stream),
      w_bar(m_c, stream),
      omega(m_c, stream),
      rho(K_in, stream),
      element_cone_ids(m_c, stream),
      scratch(K_in, stream)
  {
    std::vector<i_t> offsets(K + 1, 0);
    std::vector<i_t> blk_offsets(K + 1, 0);
    std::vector<i_t> cone_ids(m_c, 0);
    std::vector<i_t> block_cone_ids(block_entry_cone_ids.size(), 0);

    for (i_t i = 0; i < K; ++i) {
      offsets[i + 1]     = offsets[i] + dims[i];
      blk_offsets[i + 1] = blk_offsets[i] + dims[i] * dims[i];
      std::fill(cone_ids.begin() + offsets[i], cone_ids.begin() + offsets[i + 1], i);
      std::fill(
        block_cone_ids.begin() + blk_offsets[i], block_cone_ids.begin() + blk_offsets[i + 1], i);
    }

    auto init_device_vec = [&](auto& d_vec, const auto& h_vec) {
      if (!h_vec.empty()) {
        d_vec.resize(h_vec.size(), stream);
        raft::copy(d_vec.data(), h_vec.data(), h_vec.size(), stream);
      }
    };

    raft::copy(cone_offsets.data(), offsets.data(), K + 1, stream);
    raft::copy(cone_dims.data(), dims.data(), K, stream);
    raft::copy(block_offsets.data(), blk_offsets.data(), K + 1, stream);
    init_device_vec(block_entry_cone_ids, block_cone_ids);
    init_device_vec(element_cone_ids, cone_ids);
  }
};

template <typename i_t, typename f_t, typename InputIt>
void segmented_sum(InputIt input,
                   raft::device_span<const i_t> cone_offsets,
                   i_t K,
                   raft::device_span<f_t> out,
                   rmm::device_uvector<std::uint8_t>& workspace,
                   rmm::cuda_stream_view stream)
{
  if (K == 0) return;
  cuopt_assert(static_cast<i_t>(out.size()) == K, "segmented_sum output must match cone count");

  std::size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(nullptr,
                                  temp_storage_bytes,
                                  input,
                                  out.data(),
                                  K,
                                  cone_offsets.data(),
                                  cone_offsets.data() + 1,
                                  stream.value());
  if (workspace.size() < temp_storage_bytes) { workspace.resize(temp_storage_bytes, stream); }
  cub::DeviceSegmentedReduce::Sum(workspace.data(),
                                  temp_storage_bytes,
                                  input,
                                  out.data(),
                                  K,
                                  cone_offsets.data(),
                                  cone_offsets.data() + 1,
                                  stream.value());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename t_t, typename InputIt, typename ReduceOp>
void segmented_reduce(InputIt input,
                      raft::device_span<const i_t> cone_offsets,
                      i_t K,
                      rmm::device_uvector<t_t>& out,
                      rmm::device_uvector<std::uint8_t>& workspace,
                      ReduceOp reduce_op,
                      t_t initial_value,
                      rmm::cuda_stream_view stream)
{
  out.resize(K, stream);
  if (K == 0) return;

  std::size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(nullptr,
                                     temp_storage_bytes,
                                     input,
                                     out.data(),
                                     K,
                                     cone_offsets.data(),
                                     cone_offsets.data() + 1,
                                     reduce_op,
                                     initial_value,
                                     stream.value());
  if (workspace.size() < temp_storage_bytes) { workspace.resize(temp_storage_bytes, stream); }
  cub::DeviceSegmentedReduce::Reduce(workspace.data(),
                                     temp_storage_bytes,
                                     input,
                                     out.data(),
                                     K,
                                     cone_offsets.data(),
                                     cone_offsets.data() + 1,
                                     reduce_op,
                                     initial_value,
                                     stream.value());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
void apply_hinv2(raft::device_span<const f_t> v,
                 raft::device_span<f_t> out,
                 raft::device_span<const f_t> w_bar,
                 raft::device_span<const f_t> inv_eta,
                 raft::device_span<const i_t> cone_offsets,
                 raft::device_span<const i_t> element_cone_ids,
                 raft::device_span<f_t> tail_dot,
                 rmm::device_uvector<std::uint8_t>& workspace,
                 i_t K,
                 rmm::cuda_stream_view stream,
                 f_t output_scale = f_t(1))
{
  if (K == 0) return;

  auto span_v                = v;
  auto span_w_bar            = w_bar;
  auto span_cone_offsets     = cone_offsets;
  auto span_element_cone_ids = element_cone_ids;
  auto tail_terms            = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(0),
    [span_v, span_w_bar, span_cone_offsets, span_element_cone_ids] HD(i_t idx) {
      i_t cone     = span_element_cone_ids[idx];
      i_t cone_off = span_cone_offsets[cone];
      return (idx == cone_off) ? f_t(0) : span_w_bar[idx] * span_v[idx];
    });
  segmented_sum<i_t, f_t>(tail_terms, cone_offsets, K, tail_dot, workspace, stream);

  i_t grid_dim = (static_cast<i_t>(out.size()) + flat_block_dim - 1) / flat_block_dim;
  apply_hinv2_write_kernel<i_t, f_t><<<grid_dim, flat_block_dim, 0, stream>>>(
    v, out, w_bar, inv_eta, tail_dot, cone_offsets, element_cone_ids, output_scale);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
void apply_hinv2(raft::device_span<const f_t> v,
                 raft::device_span<f_t> out,
                 cone_data_t<i_t, f_t>& cones,
                 rmm::cuda_stream_view stream,
                 f_t output_scale = f_t(1))
{
  apply_hinv2<i_t, f_t>(v,
                        out,
                        cuopt::make_span(cones.w_bar),
                        cuopt::make_span(cones.inv_eta),
                        cuopt::make_span(cones.cone_offsets),
                        cuopt::make_span(cones.element_cone_ids),
                        cones.scratch.hinv2_tail_dot(),
                        cones.scratch.segmented_reduce_workspace,
                        cones.K,
                        stream,
                        output_scale);
}

template <typename i_t, typename f_t>
void compute_affine_cone_rhs_term(cone_data_t<i_t, f_t>& cones,
                                  raft::device_span<f_t> out,
                                  rmm::cuda_stream_view stream,
                                  f_t output_scale = f_t(1))
{
  cuopt_assert(static_cast<i_t>(out.size()) == cones.m_c, "cone rhs span must match cone size");
  if (cones.K == 0) return;

  apply_hinv2<i_t, f_t>(cones.s, out, cones, stream, output_scale);
}

template <typename i_t, typename f_t>
void compute_combined_cone_rhs_term(raft::device_span<const f_t> dx_aff,
                                    cone_data_t<i_t, f_t>& cones,
                                    f_t sigma_mu,
                                    raft::device_span<f_t> out,
                                    rmm::cuda_stream_view stream,
                                    f_t output_scale = f_t(1))
{
  cuopt_assert(static_cast<i_t>(out.size()) == cones.m_c, "cone rhs span must match cone size");
  if (cones.K == 0) return;

  auto span_dx_aff          = dx_aff;
  auto span_w_bar           = cuopt::make_span(cones.w_bar);
  auto span_omega           = cuopt::make_span(cones.omega);
  auto span_cone_offsets    = cuopt::make_span(cones.cone_offsets);
  auto span_element_cone_id = cuopt::make_span(cones.element_cone_ids);

  auto raw_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(0),
    [span_dx_aff, span_w_bar, span_omega, span_cone_offsets, span_element_cone_id] HD(i_t idx) {
      i_t cone     = span_element_cone_id[idx];
      i_t cone_off = span_cone_offsets[cone];
      if (idx == cone_off) { return corrector_raw_t<f_t>{f_t(0), f_t(0), f_t(0)}; }
      f_t dx_aff_j = span_dx_aff[idx];
      return corrector_raw_t<f_t>{
        span_w_bar[idx] * dx_aff_j, span_omega[idx] * dx_aff_j, dx_aff_j * dx_aff_j};
    });
  segmented_reduce<i_t, corrector_raw_t<f_t>>(raw_terms,
                                              cuopt::make_span(cones.cone_offsets),
                                              cones.K,
                                              cones.scratch.corrector_raw,
                                              cones.scratch.segmented_reduce_workspace,
                                              corrector_raw_sum_t<f_t>{},
                                              corrector_raw_t<f_t>{f_t(0), f_t(0), f_t(0)},
                                              stream);

  i_t grid_dim = (cones.m_c + flat_block_dim - 1) / flat_block_dim;
  fused_corrector_write_kernel<i_t, f_t>
    <<<grid_dim, flat_block_dim, 0, stream>>>(cones.s,
                                              cones.lambda,
                                              dx_aff,
                                              cuopt::make_span(cones.omega),
                                              cuopt::make_span(cones.w_bar),
                                              cuopt::make_span(cones.inv_eta),
                                              cuopt::make_span(cones.inv_1pw0),
                                              cuopt::make_span(cones.rho),
                                              cuopt::make_span(cones.scratch.corrector_raw),
                                              out,
                                              cuopt::make_span(cones.cone_offsets),
                                              cuopt::make_span(cones.element_cone_ids),
                                              sigma_mu,
                                              output_scale);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
void recover_cone_dz_from_target(raft::device_span<const f_t> dx,
                                 cone_data_t<i_t, f_t>& cones,
                                 raft::device_span<const f_t> cone_target,
                                 rmm::device_uvector<f_t>& hinv2_dx,
                                 raft::device_span<f_t> dz,
                                 rmm::cuda_stream_view stream)
{
  hinv2_dx.resize(cones.m_c, stream);
  if (cones.K == 0) return;

  apply_hinv2<i_t, f_t>(dx, cuopt::make_span(hinv2_dx), cones, stream);

  auto span_target = cone_target;
  auto span_hinv2  = cuopt::make_span(hinv2_dx);
  auto span_dz     = dz;
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<i_t>(0),
                     cones.m_c,
                     [span_target, span_hinv2, span_dz] __device__(i_t j) {
                       span_dz[j] = span_target[j] - span_hinv2[j];
                     });
}

template <typename i_t, typename f_t>
void accumulate_cone_hinv2_matvec(raft::device_span<const f_t> x,
                                  cone_data_t<i_t, f_t>& cones,
                                  rmm::device_uvector<f_t>& hinv2_x,
                                  raft::device_span<f_t> out,
                                  rmm::cuda_stream_view stream)
{
  hinv2_x.resize(cones.m_c, stream);
  if (cones.K == 0) return;

  apply_hinv2<i_t, f_t>(x, cuopt::make_span(hinv2_x), cones, stream);

  auto span_hinv2 = cuopt::make_span(hinv2_x);
  auto span_out   = out;
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<i_t>(0),
                     cones.m_c,
                     [span_hinv2, span_out] __device__(i_t j) { span_out[j] += span_hinv2[j]; });
}

// ---------------------------------------------------------------------------
// Compute flat H^{-2} cone-block entries and scatter them into the augmented
// CSR value array.
//
// The caller provides one flat entry per dense cone-block element:
//   - `csr_indices[e]` gives the destination slot in `augmented_x`
//   - `q_values[e]` stores any pre-merged Q contribution for that slot
//
// For each flat entry we load its precomputed owning cone id, recover local
// (r, c) coordinates, evaluate H_k^{-2}(r, c), and write
//   -(H_k^{-2}(r, c) + q_values[e])
// into `augmented_x[csr_indices[e]]`.
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t>
__global__ void scatter_hinv2_into_augmented_kernel(
  raft::device_span<f_t> augmented_x,
  raft::device_span<const i_t> csr_indices,
  raft::device_span<const f_t> q_values,
  raft::device_span<const f_t> w_bar,
  raft::device_span<const f_t> inv_eta,
  raft::device_span<const i_t> cone_offsets,
  raft::device_span<const i_t> block_offsets,
  raft::device_span<const i_t> block_entry_cone_ids)
{
  i_t e = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= static_cast<i_t>(csr_indices.size())) return;

  i_t cone    = block_entry_cone_ids[e];
  i_t off     = cone_offsets[cone];
  i_t q       = cone_offsets[cone + 1] - off;
  i_t blk_off = block_offsets[cone];
  i_t local   = e - blk_off;
  i_t r       = local / q;
  i_t c       = local % q;

  f_t ie_sq           = inv_eta[cone] * inv_eta[cone];
  f_t w0              = w_bar[off];
  f_t u_r             = (r == 0) ? w0 : -w_bar[off + r];
  f_t u_c             = (c == 0) ? w0 : -w_bar[off + c];
  f_t val             = f_t(2) * u_r * ie_sq * u_c;
  f_t diag_correction = (r == 0) ? -ie_sq : ie_sq;
  if (r == c) { val += diag_correction; }

  augmented_x[csr_indices[e]] = -val - q_values[e];
}

template <typename i_t, typename f_t>
void scatter_hinv2_into_augmented(const cone_data_t<i_t, f_t>& cones,
                                  rmm::device_uvector<f_t>& augmented_x,
                                  const rmm::device_uvector<i_t>& csr_indices,
                                  const rmm::device_uvector<f_t>& q_values,
                                  rmm::cuda_stream_view stream)
{
  i_t count = static_cast<i_t>(csr_indices.size());
  if (count == 0) return;

  cuopt_assert(count == static_cast<i_t>(cones.block_entry_cone_ids.size()),
               "scatter expects one flat entry per cone-block coefficient");

  i_t grid_dim = (count + flat_block_dim - 1) / flat_block_dim;
  scatter_hinv2_into_augmented_kernel<i_t, f_t>
    <<<grid_dim, flat_block_dim, 0, stream>>>(cuopt::make_span(augmented_x),
                                              cuopt::make_span(csr_indices),
                                              cuopt::make_span(q_values),
                                              cuopt::make_span(cones.w_bar),
                                              cuopt::make_span(cones.inv_eta),
                                              cuopt::make_span(cones.cone_offsets),
                                              cuopt::make_span(cones.block_offsets),
                                              cuopt::make_span(cones.block_entry_cone_ids));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------
// Compute per-cone step lengths, then reduce them to the global maximum
// feasible primal/dual step.
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t>
void compute_cone_step_length_per_cone(cone_data_t<i_t, f_t>& cones,
                                       raft::device_span<const f_t> x_K,
                                       raft::device_span<const f_t> dx_K,
                                       raft::device_span<const f_t> z_K,
                                       raft::device_span<const f_t> dz_K,
                                       raft::device_span<f_t> alpha_primal,
                                       raft::device_span<f_t> alpha_dual,
                                       f_t alpha_max,
                                       rmm::cuda_stream_view stream)
{
  cuopt_assert(static_cast<i_t>(alpha_primal.size()) == cones.K &&
                 static_cast<i_t>(alpha_dual.size()) == cones.K,
               "step-length outputs must match cone count");
  if (cones.K == 0) return;

  auto span_offsets = cuopt::make_span(cones.cone_offsets);
  auto span_elem    = cuopt::make_span(cones.element_cone_ids);

  auto s_du1_sq = cones.scratch.step_s_du1_sq();
  auto s_u1du1  = cones.scratch.step_s_u1du1();
  auto s_u1_sq  = cones.scratch.step_s_u1_sq();
  auto l_du1_sq = cones.scratch.step_l_du1_sq();
  auto l_u1du1  = cones.scratch.step_l_u1du1();
  auto l_u1_sq  = cones.scratch.step_l_u1_sq();

  auto span_x_K  = x_K;
  auto span_dx_K = dx_K;
  auto span_z_K  = z_K;
  auto span_dz_K = dz_K;

  auto s_du1_sq_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(0), [span_dx_K, span_offsets, span_elem] HD(i_t idx) {
      i_t cone = span_elem[idx];
      return (idx == span_offsets[cone]) ? f_t(0) : span_dx_K[idx] * span_dx_K[idx];
    });
  segmented_sum<i_t, f_t>(s_du1_sq_terms,
                          span_offsets,
                          cones.K,
                          s_du1_sq,
                          cones.scratch.segmented_reduce_workspace,
                          stream);

  auto s_u1du1_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(0),
    [span_x_K, span_dx_K, span_offsets, span_elem] HD(i_t idx) {
      i_t cone = span_elem[idx];
      return (idx == span_offsets[cone]) ? f_t(0) : span_x_K[idx] * span_dx_K[idx];
    });
  segmented_sum<i_t, f_t>(s_u1du1_terms,
                          span_offsets,
                          cones.K,
                          s_u1du1,
                          cones.scratch.segmented_reduce_workspace,
                          stream);

  auto s_u1_sq_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(0), [span_x_K, span_offsets, span_elem] HD(i_t idx) {
      i_t cone = span_elem[idx];
      return (idx == span_offsets[cone]) ? f_t(0) : span_x_K[idx] * span_x_K[idx];
    });
  segmented_sum<i_t, f_t>(s_u1_sq_terms,
                          span_offsets,
                          cones.K,
                          s_u1_sq,
                          cones.scratch.segmented_reduce_workspace,
                          stream);

  auto l_du1_sq_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(0), [span_dz_K, span_offsets, span_elem] HD(i_t idx) {
      i_t cone = span_elem[idx];
      return (idx == span_offsets[cone]) ? f_t(0) : span_dz_K[idx] * span_dz_K[idx];
    });
  segmented_sum<i_t, f_t>(l_du1_sq_terms,
                          span_offsets,
                          cones.K,
                          l_du1_sq,
                          cones.scratch.segmented_reduce_workspace,
                          stream);

  auto l_u1du1_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(0),
    [span_z_K, span_dz_K, span_offsets, span_elem] HD(i_t idx) {
      i_t cone = span_elem[idx];
      return (idx == span_offsets[cone]) ? f_t(0) : span_z_K[idx] * span_dz_K[idx];
    });
  segmented_sum<i_t, f_t>(l_u1du1_terms,
                          span_offsets,
                          cones.K,
                          l_u1du1,
                          cones.scratch.segmented_reduce_workspace,
                          stream);

  auto l_u1_sq_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(0), [span_z_K, span_offsets, span_elem] HD(i_t idx) {
      i_t cone = span_elem[idx];
      return (idx == span_offsets[cone]) ? f_t(0) : span_z_K[idx] * span_z_K[idx];
    });
  segmented_sum<i_t, f_t>(l_u1_sq_terms,
                          span_offsets,
                          cones.K,
                          l_u1_sq,
                          cones.scratch.segmented_reduce_workspace,
                          stream);

  i_t grid_dim = (cones.K + flat_block_dim - 1) / flat_block_dim;
  step_length_pair_kernel<i_t, f_t><<<grid_dim, flat_block_dim, 0, stream>>>(x_K,
                                                                             dx_K,
                                                                             z_K,
                                                                             dz_K,
                                                                             alpha_primal,
                                                                             alpha_dual,
                                                                             s_du1_sq,
                                                                             s_u1du1,
                                                                             s_u1_sq,
                                                                             l_du1_sq,
                                                                             l_u1du1,
                                                                             l_u1_sq,
                                                                             span_offsets,
                                                                             alpha_max,
                                                                             cones.K);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
std::pair<f_t, f_t> compute_cone_step_length(cone_data_t<i_t, f_t>& cones,
                                             raft::device_span<const f_t> x_K,
                                             raft::device_span<const f_t> dx_K,
                                             raft::device_span<const f_t> z_K,
                                             raft::device_span<const f_t> dz_K,
                                             f_t alpha_max,
                                             rmm::cuda_stream_view stream)
{
  if (cones.K == 0) return {alpha_max, alpha_max};

  auto alpha_primal = cones.scratch.step_alpha_primal_span();
  auto alpha_dual   = cones.scratch.step_alpha_dual_span();

  compute_cone_step_length_per_cone<i_t, f_t>(
    cones, x_K, dx_K, z_K, dz_K, alpha_primal, alpha_dual, alpha_max, stream);

  f_t primal = thrust::reduce(rmm::exec_policy(stream),
                              alpha_primal.begin(),
                              alpha_primal.end(),
                              alpha_max,
                              thrust::minimum<f_t>());
  f_t dual   = thrust::reduce(rmm::exec_policy(stream),
                            alpha_dual.begin(),
                            alpha_dual.end(),
                            alpha_max,
                            thrust::minimum<f_t>());
  return {primal, dual};
}

template <typename i_t, typename f_t>
void launch_nt_scaling(cone_data_t<i_t, f_t>& cones, rmm::cuda_stream_view stream)
{
  if (cones.K == 0) return;

  auto nt_s1_sq = cones.scratch.nt_s1_sq();
  auto nt_l1_sq = cones.scratch.nt_l1_sq();
  auto nt_sl    = cones.scratch.nt_sl();

  auto span_s       = cones.s;
  auto span_lambda  = cones.lambda;
  auto span_offsets = cuopt::make_span(cones.cone_offsets);
  auto span_elem    = cuopt::make_span(cones.element_cone_ids);

  auto s1_sq_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(0), [span_s, span_offsets, span_elem] HD(i_t idx) {
      i_t cone = span_elem[idx];
      return (idx == span_offsets[cone]) ? f_t(0) : span_s[idx] * span_s[idx];
    });
  segmented_sum<i_t, f_t>(
    s1_sq_terms, span_offsets, cones.K, nt_s1_sq, cones.scratch.segmented_reduce_workspace, stream);

  auto l1_sq_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(0), [span_lambda, span_offsets, span_elem] HD(i_t idx) {
      i_t cone = span_elem[idx];
      return (idx == span_offsets[cone]) ? f_t(0) : span_lambda[idx] * span_lambda[idx];
    });
  segmented_sum<i_t, f_t>(
    l1_sq_terms, span_offsets, cones.K, nt_l1_sq, cones.scratch.segmented_reduce_workspace, stream);

  auto sl_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(0),
    [span_s, span_lambda, span_offsets, span_elem] HD(i_t idx) {
      i_t cone = span_elem[idx];
      return (idx == span_offsets[cone]) ? f_t(0) : span_s[idx] * span_lambda[idx];
    });
  segmented_sum<i_t, f_t>(
    sl_terms, span_offsets, cones.K, nt_sl, cones.scratch.segmented_reduce_workspace, stream);

  i_t scalar_grid_dim = (cones.K + flat_block_dim - 1) / flat_block_dim;
  nt_scaling_scalar_kernel<i_t, f_t>
    <<<scalar_grid_dim, flat_block_dim, 0, stream>>>(cones.s,
                                                     cones.lambda,
                                                     span_offsets,
                                                     nt_s1_sq,
                                                     nt_l1_sq,
                                                     nt_sl,
                                                     cuopt::make_span(cones.inv_eta),
                                                     cuopt::make_span(cones.inv_1pw0),
                                                     cuopt::make_span(cones.w_bar),
                                                     cuopt::make_span(cones.omega),
                                                     cuopt::make_span(cones.rho),
                                                     cones.K);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  i_t grid_dim = (cones.m_c + flat_block_dim - 1) / flat_block_dim;
  nt_scaling_tail_kernel<i_t, f_t>
    <<<grid_dim, flat_block_dim, 0, stream>>>(cones.s,
                                              cones.lambda,
                                              cuopt::make_span(cones.inv_eta),
                                              cuopt::make_span(cones.rho),
                                              cuopt::make_span(cones.w_bar),
                                              cuopt::make_span(cones.omega),
                                              span_offsets,
                                              span_elem);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuopt::linear_programming::dual_simplex
