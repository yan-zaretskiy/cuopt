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

#include <raft/util/cudart_utils.hpp>

#include <cuda/std/tuple>

#include <cub/block/block_reduce.cuh>

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// ---------------------------------------------------------------------------
// Shared reduction primitives
// ---------------------------------------------------------------------------

template <typename f_t>
using triplet_t = cuda::std::tuple<f_t, f_t, f_t>;

template <typename f_t>
struct triplet_sum {
  DI triplet_t<f_t> operator()(const triplet_t<f_t>& lhs, const triplet_t<f_t>& rhs) const
  {
    const auto& [v0_l, v1_l, v2_l] = lhs;
    const auto& [v0_r, v1_r, v2_r] = rhs;
    return {v0_l + v0_r, v1_l + v1_r, v2_l + v2_r};
  }
};

template <typename T, int BLOCK_DIM>
using block_reduce_t = cub::BlockReduce<T, BLOCK_DIM, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

template <typename f_t, int BLOCK_DIM>
struct smem_reduce_t {
  using ScalarReduce  = block_reduce_t<f_t, BLOCK_DIM>;
  using TripletReduce = block_reduce_t<triplet_t<f_t>, BLOCK_DIM>;

  union {
    typename ScalarReduce::TempStorage scalar_temp;
    typename TripletReduce::TempStorage triplet_temp;
    f_t scalar_broadcast;
    triplet_t<f_t> triplet_broadcast;
  };
};

// ---------------------------------------------------------------------------
// reduce_broadcast: block-reduce a value, then broadcast to all threads.
// ---------------------------------------------------------------------------

template <typename f_t, int BLOCK_DIM>
DI f_t reduce_broadcast(f_t val, smem_reduce_t<f_t, BLOCK_DIM>& s)
{
  f_t agg = typename smem_reduce_t<f_t, BLOCK_DIM>::ScalarReduce(s.scalar_temp).Sum(val);
  __syncthreads();
  if (threadIdx.x == 0) { s.scalar_broadcast = agg; }
  __syncthreads();
  return s.scalar_broadcast;
}

template <typename f_t, int BLOCK_DIM>
DI triplet_t<f_t> reduce_broadcast(triplet_t<f_t> val, smem_reduce_t<f_t, BLOCK_DIM>& s)
{
  auto agg = typename smem_reduce_t<f_t, BLOCK_DIM>::TripletReduce(s.triplet_temp)
               .Reduce(val, triplet_sum<f_t>{});
  __syncthreads();
  if (threadIdx.x == 0) { s.triplet_broadcast = agg; }
  __syncthreads();
  return s.triplet_broadcast;
}

template <typename f_t, int BLOCK_DIM>
struct smem_warp_reduce_t {
  static constexpr int warps_per_block = BLOCK_DIM / 32;

  using ScalarReduce  = cub::WarpReduce<f_t, 32>;
  using TripletReduce = cub::WarpReduce<triplet_t<f_t>, 32>;

  union {
    typename ScalarReduce::TempStorage scalar_temp[warps_per_block];
    typename TripletReduce::TempStorage triplet_temp[warps_per_block];
    f_t scalar_broadcast[warps_per_block];
    triplet_t<f_t> triplet_broadcast[warps_per_block];
  };
};

// ---------------------------------------------------------------------------
// reduce_broadcast: warp-reduce a value, then broadcast within the warp.
// ---------------------------------------------------------------------------

template <typename f_t, int BLOCK_DIM>
DI f_t reduce_broadcast(f_t val, smem_warp_reduce_t<f_t, BLOCK_DIM>& s)
{
  static_assert(BLOCK_DIM % 32 == 0, "Warp reduce requires warp-aligned CTAs");

  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  f_t agg = typename smem_warp_reduce_t<f_t, BLOCK_DIM>::ScalarReduce(s.scalar_temp[warp]).Sum(val);
  if (lane == 0) { s.scalar_broadcast[warp] = agg; }
  __syncwarp();
  return s.scalar_broadcast[warp];
}

template <typename f_t, int BLOCK_DIM>
DI triplet_t<f_t> reduce_broadcast(triplet_t<f_t> val, smem_warp_reduce_t<f_t, BLOCK_DIM>& s)
{
  static_assert(BLOCK_DIM % 32 == 0, "Warp reduce requires warp-aligned CTAs");

  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  auto agg = typename smem_warp_reduce_t<f_t, BLOCK_DIM>::TripletReduce(s.triplet_temp[warp])
               .Reduce(val, triplet_sum<f_t>{});
  if (lane == 0) { s.triplet_broadcast[warp] = agg; }
  __syncwarp();
  return s.triplet_broadcast[warp];
}

// ---------------------------------------------------------------------------
// Apply H^{-1} to one vector per cone (one thread-block per cone).
//
// H^{-1}z = (1/η)(w̄₀z₀ − ζ,  z₁ + (−z₀ + ζ/(1+w̄₀))w̄₁),  ζ = w̄₁ᵀz₁
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t, int BLOCK_DIM>
__global__ __launch_bounds__(BLOCK_DIM) void apply_Hinv_kernel(const f_t* __restrict__ z,
                                                               f_t* __restrict__ out,
                                                               const f_t* __restrict__ w_bar,
                                                               const f_t* __restrict__ inv_eta,
                                                               const f_t* __restrict__ inv_1pw0,
                                                               const i_t* __restrict__ cone_offsets,
                                                               i_t K)
{
  __shared__ smem_reduce_t<f_t, BLOCK_DIM> smem;

  i_t cone = static_cast<i_t>(blockIdx.x);
  if (cone >= K) return;

  i_t off           = cone_offsets[cone];
  i_t q             = cone_offsets[cone + 1] - off;
  const f_t* w_cone = w_bar + off;
  const f_t* z_cone = z + off;
  f_t* out_cone     = out + off;

  f_t z0 = z_cone[0];
  f_t w0 = w_cone[0];

  // Phase 1: ζ = w̄₁ᵀ z₁
  f_t partial = f_t(0);
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    partial += w_cone[j] * z_cone[j];
  }
  f_t zeta = reduce_broadcast(partial, smem);

  // Phase 2: element-wise output
  f_t ie    = inv_eta[cone];
  f_t ipw   = inv_1pw0[cone];
  f_t coeff = -z0 + zeta * ipw;

  if (threadIdx.x == 0) { out_cone[0] = (w0 * z0 - zeta) * ie; }
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    out_cone[j] = (z_cone[j] + coeff * w_cone[j]) * ie;
  }
}

// ---------------------------------------------------------------------------
// Apply H^{-2} to one vector per cone (one thread-block per cone).
//
// H^{-2}v = η⁻²(2u(uᵀv) − Jv),   u = Jw̄,   J = diag(1,−1,…,−1).
//
// One dot product (uᵀv) plus element-wise work — same structure as apply_Hinv.
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t, int BLOCK_DIM>
__global__ __launch_bounds__(BLOCK_DIM) void apply_Hinv2_kernel(
  const f_t* __restrict__ v,
  f_t* __restrict__ out,
  const f_t* __restrict__ w_bar,
  const f_t* __restrict__ inv_eta,
  const i_t* __restrict__ cone_offsets,
  i_t K)
{
  __shared__ smem_reduce_t<f_t, BLOCK_DIM> smem;

  i_t cone = static_cast<i_t>(blockIdx.x);
  if (cone >= K) return;

  i_t off           = cone_offsets[cone];
  i_t q             = cone_offsets[cone + 1] - off;
  const f_t* w_cone = w_bar + off;
  const f_t* v_cone = v + off;
  f_t* out_cone     = out + off;

  f_t v0 = v_cone[0];
  f_t w0 = w_cone[0];

  // Phase 1: uᵀv = w̄₀v₀ − Σ w̄_j v_j  (tail dot, then subtract from head)
  f_t partial = f_t(0);
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    partial += w_cone[j] * v_cone[j];
  }
  f_t tail_dot = reduce_broadcast(partial, smem);
  f_t uTv      = w0 * v0 - tail_dot;

  // Phase 2: element-wise output
  f_t ie_sq = inv_eta[cone] * inv_eta[cone];
  f_t coeff = f_t(2) * uTv * ie_sq;

  if (threadIdx.x == 0) { out_cone[0] = coeff * w0 - ie_sq * v0; }
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    out_cone[j] = -coeff * w_cone[j] + ie_sq * v_cone[j];
  }
}

// ---------------------------------------------------------------------------
// Cone-algebra primitives for the deferred combined-step corrector:
//   r_K = omega circ omega + dx_scaled circ dz_scaled - sigma mu e
//   corr = omega \ r_K
//   t_K = H^{-1} corr
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Jordan product for packed SOC vectors (one CTA per cone).
//
// For a, b in Q^q:   (a ∘ b)_0 = a^T b,   (a ∘ b)_j = a_0 b_j + b_0 a_j.
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t, int BLOCK_DIM>
__global__ __launch_bounds__(BLOCK_DIM) void jordan_product_kernel(
  const f_t* __restrict__ a,
  const f_t* __restrict__ b,
  f_t* __restrict__ out,
  const i_t* __restrict__ cone_offsets,
  i_t K)
{
  __shared__ smem_reduce_t<f_t, BLOCK_DIM> smem;

  i_t cone = static_cast<i_t>(blockIdx.x);
  if (cone >= K) return;

  i_t off           = cone_offsets[cone];
  i_t q             = cone_offsets[cone + 1] - off;
  const f_t* a_cone = a + off;
  const f_t* b_cone = b + off;
  f_t* out_cone     = out + off;

  f_t a0 = a_cone[0];
  f_t b0 = b_cone[0];

  f_t partial = f_t(0);
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    partial += a_cone[j] * b_cone[j];
  }
  f_t tail_dot = reduce_broadcast(partial, smem);

  if (threadIdx.x == 0) { out_cone[0] = a0 * b0 + tail_dot; }

  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    out_cone[j] = a0 * b_cone[j] + b0 * a_cone[j];
  }
}

// ---------------------------------------------------------------------------
// Inverse Jordan product for packed SOC vectors (one CTA per cone).
//
// For omega in int(Q^q) and vector r,
//   (omega \ r)_0 = (omega_0 r_0 − nu) / rho
//   (omega \ r)_j = ((nu/omega_0 − r_0)/rho) omega_j + r_j/omega_0
// where nu = omega_1^T r_1 and rho = ||omega||_J^2 (stored per-cone).
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t, int BLOCK_DIM>
__global__ __launch_bounds__(BLOCK_DIM) void inverse_jordan_product_kernel(
  const f_t* __restrict__ omega,
  const f_t* __restrict__ r,
  const f_t* __restrict__ rho,
  f_t* __restrict__ out,
  const i_t* __restrict__ cone_offsets,
  i_t K)
{
  __shared__ smem_reduce_t<f_t, BLOCK_DIM> smem;

  i_t cone = static_cast<i_t>(blockIdx.x);
  if (cone >= K) return;

  i_t off               = cone_offsets[cone];
  i_t q                 = cone_offsets[cone + 1] - off;
  const f_t* omega_cone = omega + off;
  const f_t* r_cone     = r + off;
  f_t* out_cone         = out + off;

  f_t omega_0 = omega_cone[0];
  f_t r_0     = r_cone[0];

  f_t partial = f_t(0);
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    partial += omega_cone[j] * r_cone[j];
  }
  f_t nu = reduce_broadcast(partial, smem);

  f_t rho_val   = rho[cone];
  f_t inv_rho   = f_t(1) / rho_val;
  f_t c_omega_j = ((nu / omega_0) - r_0) * inv_rho;
  f_t c_r_j     = f_t(1) / omega_0;

  if (threadIdx.x == 0) { out_cone[0] = (omega_0 * r_0 - nu) * inv_rho; }
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    out_cone[j] = c_omega_j * omega_cone[j] + c_r_j * r_cone[j];
  }
}

// ---------------------------------------------------------------------------
// Fused corrector for the combined-step SOC correction (one CTA per cone).
//
// Computes in a single kernel launch:
//   1. dx  = H^{-1} Δx_aff                      (affine scaled direction)
//   2. dz  = −ω − dx                             (complementary direction)
//   3. r_K = ω∘ω + dx∘dz − σμ e                  (combined cone residual)
//   4. corr = ω \ r_K                            (inverse Jordan product)
//   5. t_K = H^{-1} corr                         (corrector for reduced RHS)
//
// Uses the `out` buffer as scratch (holds dx during phases 1–3) and writes
// the final t_K there, so zero extra temporary buffers are needed.
//
// Algebraic shortcut:  the triplet (Σ ω_j², Σ ω_j dx_j, Σ dx_j²) computed
// for r_K_0 also yields ν = Σ ω_j r_K_j via a linear combination, avoiding
// a fourth reduction pass.
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t, int BLOCK_DIM>
__global__ __launch_bounds__(BLOCK_DIM) void fused_corrector_kernel(
  const f_t* __restrict__ dx_aff,
  const f_t* __restrict__ omega,
  const f_t* __restrict__ w_bar,
  const f_t* __restrict__ inv_eta,
  const f_t* __restrict__ inv_1pw0,
  const f_t* __restrict__ rho,
  f_t sigma_mu,
  f_t* __restrict__ out,
  const i_t* __restrict__ cone_offsets,
  i_t K)
{
  __shared__ smem_reduce_t<f_t, BLOCK_DIM> smem;

  i_t cone = static_cast<i_t>(blockIdx.x);
  if (cone >= K) return;

  i_t off               = cone_offsets[cone];
  i_t q                 = cone_offsets[cone + 1] - off;
  const f_t* dx_a       = dx_aff + off;
  const f_t* omega_cone = omega + off;
  const f_t* w_cone     = w_bar + off;
  f_t* out_cone         = out + off;

  f_t ie      = inv_eta[cone];
  f_t ipw     = inv_1pw0[cone];
  f_t rho_val = rho[cone];
  f_t omega_0 = omega_cone[0];
  f_t w_0     = w_cone[0];
  f_t dx_a_0  = dx_a[0];

  // =================================================================
  // Phase A — reduce ζ = Σ_{j≥1} w̄_j (Δx_aff)_j  for H^{-1}
  // =================================================================
  f_t partial = f_t(0);
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    partial += w_cone[j] * dx_a[j];
  }
  f_t zeta = reduce_broadcast(partial, smem);

  f_t dx_0    = (w_0 * dx_a_0 - zeta) * ie;
  f_t coeff_a = -dx_a_0 + zeta * ipw;
  f_t dz_0    = -omega_0 - dx_0;

  // =================================================================
  // Phase A→B — write dx to out; accumulate (A, B, C) for r_K and ν
  //   A = Σ ω_j²,  B = Σ ω_j dx_j,  C = Σ dx_j²   (j ≥ 1)
  // =================================================================
  auto trip             = triplet_t<f_t>{};
  auto& [A_p, B_p, C_p] = trip;
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    f_t dx_j    = (dx_a[j] + coeff_a * w_cone[j]) * ie;
    out_cone[j] = dx_j;
    f_t omega_j = omega_cone[j];
    A_p += omega_j * omega_j;
    B_p += omega_j * dx_j;
    C_p += dx_j * dx_j;
  }
  auto [A, B, C] = reduce_broadcast(trip, smem);

  // =================================================================
  // Phase B — form r_K_0, derive ν, then inverse-Jordan scalars
  // =================================================================
  f_t r_K_0 = (omega_0 * omega_0 + A) + (dx_0 * dz_0 - B - C) - sigma_mu;
  f_t nu    = (f_t(2) * omega_0 - dx_0) * A - (omega_0 + f_t(2) * dx_0) * B;

  f_t inv_rho     = f_t(1) / rho_val;
  f_t corr_0      = (omega_0 * r_K_0 - nu) * inv_rho;
  f_t inv_omega_0 = f_t(1) / omega_0;
  f_t c_inv       = (nu * inv_omega_0 - r_K_0) * inv_rho;
  f_t p1          = c_inv + f_t(2) - dx_0 * inv_omega_0;
  f_t p2          = -(f_t(1) + f_t(2) * dx_0 * inv_omega_0);

  // =================================================================
  // Phase B→C — accumulate ζ₂ = Σ_{j≥1} w̄_j corr_j  for final H^{-1}
  //   corr_j = p1 ω_j + p2 dx_j   (dx_j still in out_cone[j])
  // =================================================================
  f_t partial2 = f_t(0);
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    f_t corr_j = p1 * omega_cone[j] + p2 * out_cone[j];
    partial2 += w_cone[j] * corr_j;
  }
  f_t zeta2 = reduce_broadcast(partial2, smem);

  // =================================================================
  // Phase C — write t_K = H^{-1}(corr)
  // =================================================================
  f_t coeff_c = -corr_0 + zeta2 * ipw;

  if (threadIdx.x == 0) { out_cone[0] = (w_0 * corr_0 - zeta2) * ie; }
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    f_t corr_j  = p1 * omega_cone[j] + p2 * out_cone[j];
    out_cone[j] = (corr_j + coeff_c * w_cone[j]) * ie;
  }
}

// ---------------------------------------------------------------------------
// Compute NT scaling from (s, lambda).
//
// Medium/large cones use one CTA per cone and stream s/lambda twice:
//   Pass 1: reduce ||s_1||^2, ||lambda_1||^2, and s^T lambda.
//   Pass 2: compute omega/w_bar directly from raw inputs and reduce ||w_bar_1||^2.
//
// Small cones (q <= 32) use one warp per cone and keep one element per lane in
// registers for the whole computation. In both paths, shared memory only stores
// per-warp partial reductions plus a small scalar broadcast struct.
// ---------------------------------------------------------------------------

constexpr int small_cone_limit  = 32;
constexpr int medium_cone_limit = 2048;
constexpr int small_block_dim   = 64;
constexpr int medium_block_dim  = 128;
constexpr int large_block_dim   = 256;

template <typename f_t>
struct nt_broadcast_coeffs {
  f_t w_from_s;
  f_t w_from_lambda;
  f_t omega_s_coeff;
  f_t omega_lambda_coeff;
};

template <typename f_t, int BLOCK_DIM>
struct nt_block_storage {
  smem_reduce_t<f_t, BLOCK_DIM> reduce;
  nt_broadcast_coeffs<f_t> coeffs;
};

template <typename f_t, int BLOCK_DIM>
struct nt_warp_storage {
  static constexpr int warps_per_block = BLOCK_DIM / 32;

  smem_warp_reduce_t<f_t, BLOCK_DIM> reduce;
  nt_broadcast_coeffs<f_t> coeffs[warps_per_block];
};

template <typename i_t, typename f_t, int BLOCK_DIM>
__global__ __launch_bounds__(BLOCK_DIM) void nt_scaling_kernel(const f_t* __restrict__ s,
                                                               const f_t* __restrict__ lambda,
                                                               f_t* __restrict__ eta,
                                                               f_t* __restrict__ inv_eta,
                                                               f_t* __restrict__ inv_1pw0,
                                                               f_t* __restrict__ w_bar,
                                                               f_t* __restrict__ omega,
                                                               f_t* __restrict__ rho,
                                                               const i_t* __restrict__ cone_offsets,
                                                               const i_t* __restrict__ cone_ids,
                                                               i_t num_cones)
{
  static_assert(BLOCK_DIM % 32 == 0, "NT scaling kernel requires warp-aligned BLOCK_DIM");
  __shared__ nt_block_storage<f_t, BLOCK_DIM> storage;

  i_t cone_idx = static_cast<i_t>(blockIdx.x);
  if (cone_idx >= num_cones) return;

  i_t cone = cone_ids[cone_idx];
  i_t off  = cone_offsets[cone];
  i_t q    = cone_offsets[cone + 1] - off;

  f_t s0 = s[off];
  f_t l0 = lambda[off];

  auto partial                   = triplet_t<f_t>{};
  auto& [s1_sq_p, l1_sq_p, sl_p] = partial;
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    f_t sj = s[off + j];
    f_t lj = lambda[off + j];
    s1_sq_p += sj * sj;
    l1_sq_p += lj * lj;
    sl_p += sj * lj;
  }

  auto [s1_sq, l1_sq, sl] = reduce_broadcast(partial, storage.reduce);
  f_t owner_eta           = f_t(0);
  f_t owner_inv_eta       = f_t(0);
  f_t owner_rho           = f_t(0);
  f_t owner_omega_0       = f_t(0);
  if (threadIdx.x == 0) {
    // Clamp radicands to zero: near the cone boundary, roundoff can make these
    // slightly negative.
    f_t s_J       = sqrt(max(f_t(0), s0 * s0 - s1_sq));
    f_t l_J       = sqrt(max(f_t(0), l0 * l0 - l1_sq));
    f_t inv_s_J   = f_t(1) / s_J;
    f_t inv_l_J   = f_t(1) / l_J;
    owner_rho     = s_J * l_J;
    owner_eta     = sqrt(s_J / l_J);
    owner_inv_eta = f_t(1) / owner_eta;
    f_t scale     = sqrt(owner_rho);

    f_t s_dot_l = (s0 * l0 + sl) * inv_s_J * inv_l_J;
    f_t gamma   = sqrt(max(f_t(0), (f_t(1) + s_dot_l) * f_t(0.5)));
    f_t inv_2g  = f_t(1) / (f_t(2) * gamma);
    f_t sb0     = s0 * inv_s_J;
    f_t lb0     = l0 * inv_l_J;
    f_t D       = sb0 + lb0 + f_t(2) * gamma;
    f_t inv_D   = f_t(1) / D;
    f_t c_s     = (gamma + sb0) * inv_D;
    f_t c_l     = (gamma + lb0) * inv_D;

    storage.coeffs.w_from_s      = inv_2g * inv_s_J;
    storage.coeffs.w_from_lambda = -inv_2g * inv_l_J;
    // Name these by the raw tail element they multiply:
    // omega_j = omega_s_coeff * s_j + omega_lambda_coeff * lambda_j.
    // The closed-form NT expression is cross-coupled, so c_l multiplies s_j
    // and c_s multiplies lambda_j.
    storage.coeffs.omega_s_coeff      = scale * c_l * inv_s_J;
    storage.coeffs.omega_lambda_coeff = scale * c_s * inv_l_J;
    owner_omega_0                     = gamma * scale;
  }
  __syncthreads();

  f_t w1_sq_partial = f_t(0);
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    f_t sj         = s[off + j];
    f_t lj         = lambda[off + j];
    f_t wj         = storage.coeffs.w_from_s * sj + storage.coeffs.w_from_lambda * lj;
    w_bar[off + j] = wj;
    omega[off + j] = storage.coeffs.omega_s_coeff * sj + storage.coeffs.omega_lambda_coeff * lj;
    w1_sq_partial += wj * wj;
  }

  f_t w1_sq = reduce_broadcast(w1_sq_partial, storage.reduce);
  if (threadIdx.x == 0) {
    f_t w0         = sqrt(f_t(1) + w1_sq);
    omega[off]     = owner_omega_0;
    w_bar[off]     = w0;
    eta[cone]      = owner_eta;
    inv_eta[cone]  = owner_inv_eta;
    inv_1pw0[cone] = f_t(1) / (f_t(1) + w0);
    rho[cone]      = owner_rho;
  }
}

template <typename i_t, typename f_t, int BLOCK_DIM>
__global__ __launch_bounds__(BLOCK_DIM) void nt_scaling_small_kernel(
  const f_t* __restrict__ s,
  const f_t* __restrict__ lambda,
  f_t* __restrict__ eta,
  f_t* __restrict__ inv_eta,
  f_t* __restrict__ inv_1pw0,
  f_t* __restrict__ w_bar,
  f_t* __restrict__ omega,
  f_t* __restrict__ rho,
  const i_t* __restrict__ cone_offsets,
  const i_t* __restrict__ cone_ids,
  i_t num_cones)
{
  static_assert(BLOCK_DIM % 32 == 0, "Small-cone NT kernel requires warp-aligned CTAs");
  __shared__ nt_warp_storage<f_t, BLOCK_DIM> storage;

  constexpr int warps_per_block = BLOCK_DIM / 32;
  i_t warp_idx =
    static_cast<i_t>(blockIdx.x) * warps_per_block + static_cast<i_t>(threadIdx.x >> 5);
  if (warp_idx >= num_cones) return;

  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  i_t cone = cone_ids[warp_idx];
  i_t off  = cone_offsets[cone];
  i_t q    = cone_offsets[cone + 1] - off;

  f_t sj = (lane < q) ? s[off + lane] : f_t(0);
  f_t lj = (lane < q) ? lambda[off + lane] : f_t(0);

  auto partial = triplet_t<f_t>{(lane > 0 && lane < q) ? sj * sj : f_t(0),
                                (lane > 0 && lane < q) ? lj * lj : f_t(0),
                                (lane > 0 && lane < q) ? sj * lj : f_t(0)};
  auto [s1_sq, l1_sq, sl] = reduce_broadcast(partial, storage.reduce);

  f_t owner_eta     = f_t(0);
  f_t owner_inv_eta = f_t(0);
  f_t owner_rho     = f_t(0);
  f_t owner_omega_0 = f_t(0);

  if (lane == 0) {
    f_t s0        = sj;
    f_t l0        = lj;
    f_t s_J       = sqrt(max(f_t(0), s0 * s0 - s1_sq));
    f_t l_J       = sqrt(max(f_t(0), l0 * l0 - l1_sq));
    f_t inv_s_J   = f_t(1) / s_J;
    f_t inv_l_J   = f_t(1) / l_J;
    owner_rho     = s_J * l_J;
    owner_eta     = sqrt(s_J / l_J);
    owner_inv_eta = f_t(1) / owner_eta;
    f_t scale     = sqrt(owner_rho);

    f_t s_dot_l = (s0 * l0 + sl) * inv_s_J * inv_l_J;
    f_t gamma   = sqrt(max(f_t(0), (f_t(1) + s_dot_l) * f_t(0.5)));
    f_t inv_2g  = f_t(1) / (f_t(2) * gamma);
    f_t sb0     = s0 * inv_s_J;
    f_t lb0     = l0 * inv_l_J;
    f_t D       = sb0 + lb0 + f_t(2) * gamma;
    f_t inv_D   = f_t(1) / D;
    f_t c_s     = (gamma + sb0) * inv_D;
    f_t c_l     = (gamma + lb0) * inv_D;

    storage.coeffs[warp].w_from_s           = inv_2g * inv_s_J;
    storage.coeffs[warp].w_from_lambda      = -inv_2g * inv_l_J;
    storage.coeffs[warp].omega_s_coeff      = scale * c_l * inv_s_J;
    storage.coeffs[warp].omega_lambda_coeff = scale * c_s * inv_l_J;
    owner_omega_0                           = gamma * scale;
  }
  __syncwarp();

  f_t w1_sq = f_t(0);
  if (lane > 0 && lane < q) {
    f_t wj = storage.coeffs[warp].w_from_s * sj + storage.coeffs[warp].w_from_lambda * lj;
    w_bar[off + lane] = wj;
    omega[off + lane] =
      storage.coeffs[warp].omega_s_coeff * sj + storage.coeffs[warp].omega_lambda_coeff * lj;
    w1_sq = wj * wj;
  }
  w1_sq = reduce_broadcast(w1_sq, storage.reduce);

  if (lane == 0) {
    f_t w0         = sqrt(f_t(1) + w1_sq);
    omega[off]     = owner_omega_0;
    w_bar[off]     = w0;
    eta[cone]      = owner_eta;
    inv_eta[cone]  = owner_inv_eta;
    inv_1pw0[cone] = f_t(1) / (f_t(1) + w0);
    rho[cone]      = owner_rho;
  }
}

// ---------------------------------------------------------------------------
// Step length for a single (u, du) pair in Q^q.
//
// Finds the largest alpha in [0, alpha_max] such that u + alpha*du in Q^q.
// The cone condition u_0 + alpha*du_0 >= ||u_1 + alpha*du_1|| reduces to a
// linear test plus a quadratic a*alpha^2 + 2b*alpha + c >= 0 where
//   a = du_0^2 - ||du_1||^2,  b = u_0*du_0 - u_1^T du_1,  c = u_0^2 - ||u_1||^2.
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t, int BLOCK_DIM>
DI f_t
cone_step_length_single(const f_t* __restrict__ u,
                        const f_t* __restrict__ du,
                        i_t q,
                        typename block_reduce_t<triplet_t<f_t>, BLOCK_DIM>::TempStorage& temp,
                        f_t alpha)
{
  auto partial                       = triplet_t<f_t>{};
  auto& [du1_sq_p, u1du1_p, u1_sq_p] = partial;
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    f_t uj  = u[j];
    f_t duj = du[j];
    du1_sq_p += duj * duj;
    u1du1_p += uj * duj;
    u1_sq_p += uj * uj;
  }

  auto [du1_sq, u1du1, u1_sq] =
    block_reduce_t<triplet_t<f_t>, BLOCK_DIM>(temp).Reduce(partial, triplet_sum<f_t>{});
  __syncthreads();

  if (threadIdx.x == 0) {
    f_t a    = du[0] * du[0] - du1_sq;
    f_t b    = u[0] * du[0] - u1du1;
    f_t c    = max(f_t(0), u[0] * u[0] - u1_sq);
    f_t disc = b * b - a * c;

    // Linear constraint: u_0 + alpha * du_0 >= 0.
    if (du[0] < f_t(0)) { alpha = min(alpha, -u[0] / du[0]); }

    // Quadratic constraint.
    if ((a > f_t(0) && b > f_t(0)) || disc < f_t(0)) {
      // No positive root (parabola stays non-negative for alpha > 0).
    } else if (a == f_t(0)) {
      // Degenerate: 2b*alpha + c = 0.
      if (b < f_t(0)) { alpha = min(alpha, c / (f_t(-2) * b)); }
    } else if (c == f_t(0)) {
      // Starting exactly on the cone boundary: take a full step only if the
      // direction stays in the cone, otherwise the maximum feasible step is 0.
      alpha = (a >= f_t(0)) ? alpha : f_t(0);
    } else {
      f_t t  = -(b + copysign(sqrt(disc), b));
      f_t r1 = c / t;
      f_t r2 = t / a;
      if (r1 < f_t(0)) { r1 = alpha; }
      if (r2 < f_t(0)) { r2 = alpha; }
      alpha = min(alpha, min(r1, r2));
    }
  }
  return alpha;
}

// ---------------------------------------------------------------------------
// Cone step length kernel (one block per cone).
//
// Computes, for each cone i, the largest alpha in [0, alpha_max] such that
//   s_i + alpha * ds_i  in  Q^{q_i}   AND   lambda_i + alpha * dlambda_i  in  Q^{q_i}.
// The per-cone result is written to alpha[i].
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t, int BLOCK_DIM>
__global__ __launch_bounds__(BLOCK_DIM) void step_length_kernel(
  const f_t* __restrict__ s,
  const f_t* __restrict__ ds,
  const f_t* __restrict__ lambda,
  const f_t* __restrict__ dlambda,
  f_t* __restrict__ alpha,
  const i_t* __restrict__ cone_offsets,
  i_t K,
  f_t alpha_max)
{
  __shared__ typename block_reduce_t<triplet_t<f_t>, BLOCK_DIM>::TempStorage temp_storage;

  i_t cone = static_cast<i_t>(blockIdx.x);
  if (cone >= K) return;

  i_t off = cone_offsets[cone];
  i_t q   = cone_offsets[cone + 1] - off;

  f_t alpha_s =
    cone_step_length_single<i_t, f_t, BLOCK_DIM>(s + off, ds + off, q, temp_storage, alpha_max);
  f_t alpha_l = cone_step_length_single<i_t, f_t, BLOCK_DIM>(
    lambda + off, dlambda + off, q, temp_storage, alpha_max);

  if (threadIdx.x == 0) { alpha[cone] = min(alpha_s, alpha_l); }
}

// ---------------------------------------------------------------------------
// Shift u into int(Q^q) if it is not already interior (one block per cone).
//
// alpha(u) = ||u_1|| - u_0.  If alpha >= 0 (u on boundary or outside):
//   u_0 <- u_0 + 1 + max(0, alpha)     (shift along identity element e)
//
// Modifies u in place.  Used once during initial-point computation.
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t, int BLOCK_DIM>
__global__ __launch_bounds__(BLOCK_DIM) void interior_shift_kernel(
  f_t* __restrict__ u, const i_t* __restrict__ cone_offsets, i_t K)
{
  __shared__ typename block_reduce_t<f_t, BLOCK_DIM>::TempStorage temp_storage;

  i_t cone = static_cast<i_t>(blockIdx.x);
  if (cone >= K) return;

  i_t off = cone_offsets[cone];
  i_t q   = cone_offsets[cone + 1] - off;

  f_t tail_sq = f_t(0);
  for (i_t j = 1 + static_cast<i_t>(threadIdx.x); j < q; j += BLOCK_DIM) {
    f_t v = u[off + j];
    tail_sq += v * v;
  }
  tail_sq = block_reduce_t<f_t, BLOCK_DIM>(temp_storage).Sum(tail_sq);

  if (threadIdx.x == 0) {
    f_t u1_norm = sqrt(tail_sq);
    f_t gap     = u1_norm - u[off];
    if (gap >= f_t(0)) { u[off] += f_t(1) + gap; }
  }
}

/**
 * Owns device storage for second-order cone topology, iterates, and NT scaling.
 *
 * Flat arrays are packed by cone: elements [cone_offsets[i], cone_offsets[i+1])
 * belong to cone i, which has dimension cone_dims[i].
 *
 * Search directions, RHS vectors, and workspace live directly in
 * iteration_data_t (matching the existing LP/QP pattern where dx_aff, dual_rhs,
 * etc. are all top-level fields of iteration_data_t).
 */
template <typename i_t, typename f_t>
struct cone_data_t {
  // --- Topology (set once at construction) ---
  i_t K;    // number of second-order cones
  i_t m_c;  // total cone dimension = sum of cone_dims

  rmm::device_uvector<i_t> cone_offsets;  // [K+1] prefix sums of cone_dims
  rmm::device_uvector<i_t> cone_dims;     // [K]   dimension q_i of each cone

  // --- Primal/dual cone iterates (rewritten each iteration) ---
  rmm::device_uvector<f_t> s;       // [m_c] cone slack: s_i in int(Q^{q_i})
  rmm::device_uvector<f_t> lambda;  // [m_c] cone dual:  lambda_i in int(Q^{q_i})

  // --- NT scaling state (recomputed each iteration from s, lambda) ---
  rmm::device_uvector<f_t> eta;  // [K]   scaling factor eta_i = (||s_i||_J / ||lambda_i||_J)^{1/2}
  rmm::device_uvector<f_t> inv_eta;   // [K]   cached 1/eta_i
  rmm::device_uvector<f_t> inv_1pw0;  // [K]   cached 1/(1 + wbar_0_i)
  rmm::device_uvector<f_t> w_bar;     // [m_c] NT scaling direction, unit J-norm, packed by cone
  rmm::device_uvector<f_t> omega;  // [m_c] scaled variable omega_i = H_i^{-1} s_i, packed by cone
  rmm::device_uvector<f_t> rho;    // [K]   ||omega_i||^2_J = ||s_i||_J * ||lambda_i||_J
  rmm::device_uvector<i_t> small_cone_ids;   // [n_small] cone ids with q <= 32
  rmm::device_uvector<i_t> medium_cone_ids;  // [n_medium] cone ids with 32 < q <= 2048
  rmm::device_uvector<i_t> large_cone_ids;   // [n_large] cone ids with q > 2048

  cone_data_t(i_t K_in, const std::vector<i_t>& dims, rmm::cuda_stream_view stream)
    : K(K_in),
      m_c(std::accumulate(dims.begin(), dims.end(), i_t(0))),
      cone_offsets(K_in + 1, stream),
      cone_dims(K_in, stream),
      s(m_c, stream),
      lambda(m_c, stream),
      eta(K_in, stream),
      inv_eta(K_in, stream),
      inv_1pw0(K_in, stream),
      w_bar(m_c, stream),
      omega(m_c, stream),
      rho(K_in, stream),
      small_cone_ids(0, stream),
      medium_cone_ids(0, stream),
      large_cone_ids(0, stream)
  {
    std::vector<i_t> offsets(K + 1, 0);
    std::vector<i_t> small_ids;
    std::vector<i_t> medium_ids;
    std::vector<i_t> large_ids;

    for (i_t i = 0; i < K; ++i) {
      offsets[i + 1] = offsets[i] + dims[i];
      if (dims[i] <= small_cone_limit) {
        small_ids.push_back(i);
      } else if (dims[i] <= medium_cone_limit) {
        medium_ids.push_back(i);
      } else {
        large_ids.push_back(i);
      }
    }

    auto init_device_vec = [&](auto& d_vec, const auto& h_vec) {
      if (!h_vec.empty()) {
        d_vec.resize(h_vec.size(), stream);
        raft::copy(d_vec.data(), h_vec.data(), h_vec.size(), stream);
      }
    };

    raft::copy(cone_offsets.data(), offsets.data(), K + 1, stream);
    raft::copy(cone_dims.data(), dims.data(), K, stream);
    init_device_vec(small_cone_ids, small_ids);
    init_device_vec(medium_cone_ids, medium_ids);
    init_device_vec(large_cone_ids, large_ids);
  }
};

template <typename i_t, typename f_t>
void launch_nt_scaling(cone_data_t<i_t, f_t>& cones, rmm::cuda_stream_view stream)
{
  auto launch_streaming_bucket = [&](auto& cone_ids, auto block_dim_ic) {
    constexpr int block_dim = std::remove_cvref_t<decltype(block_dim_ic)>::value;
    i_t bucket_size         = static_cast<i_t>(cone_ids.size());
    if (bucket_size == 0) return;

    nt_scaling_kernel<i_t, f_t, block_dim>
      <<<bucket_size, block_dim, 0, stream>>>(cones.s.data(),
                                              cones.lambda.data(),
                                              cones.eta.data(),
                                              cones.inv_eta.data(),
                                              cones.inv_1pw0.data(),
                                              cones.w_bar.data(),
                                              cones.omega.data(),
                                              cones.rho.data(),
                                              cones.cone_offsets.data(),
                                              cone_ids.data(),
                                              bucket_size);
  };

  i_t small_count = static_cast<i_t>(cones.small_cone_ids.size());
  if (small_count > 0) {
    constexpr int warps_per_block = small_block_dim / 32;
    i_t grid_dim                  = (small_count + warps_per_block - 1) / warps_per_block;
    nt_scaling_small_kernel<i_t, f_t, small_block_dim>
      <<<grid_dim, small_block_dim, 0, stream>>>(cones.s.data(),
                                                 cones.lambda.data(),
                                                 cones.eta.data(),
                                                 cones.inv_eta.data(),
                                                 cones.inv_1pw0.data(),
                                                 cones.w_bar.data(),
                                                 cones.omega.data(),
                                                 cones.rho.data(),
                                                 cones.cone_offsets.data(),
                                                 cones.small_cone_ids.data(),
                                                 small_count);
  }

  launch_streaming_bucket(cones.medium_cone_ids, std::integral_constant<int, medium_block_dim>{});
  launch_streaming_bucket(cones.large_cone_ids, std::integral_constant<int, large_block_dim>{});
}

}  // namespace cuopt::linear_programming::dual_simplex
