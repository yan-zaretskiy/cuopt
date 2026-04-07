/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <barrier/second_order_cone.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace cuopt::linear_programming::dual_simplex::test {
namespace {

template <typename i_t>
auto build_offsets(const std::vector<i_t>& dims) -> std::vector<i_t>
{
  std::vector<i_t> offsets(dims.size() + 1, 0);
  for (std::size_t i = 0; i < dims.size(); ++i) {
    offsets[i + 1] = offsets[i] + dims[i];
  }
  return offsets;
}

template <typename f_t>
auto pack_cones(const std::vector<std::vector<f_t>>& cones) -> std::vector<f_t>
{
  std::size_t total_size = 0;
  for (const auto& cone : cones) {
    total_size += cone.size();
  }

  std::vector<f_t> packed;
  packed.reserve(total_size);
  for (const auto& cone : cones) {
    packed.insert(packed.end(), cone.begin(), cone.end());
  }
  return packed;
}

template <typename f_t, typename i_t>
auto slice_cone(const std::vector<f_t>& packed, const std::vector<i_t>& offsets, i_t cone)
  -> std::vector<f_t>
{
  auto begin = packed.begin() + offsets[cone];
  auto end   = packed.begin() + offsets[cone + 1];
  return std::vector<f_t>(begin, end);
}

template <typename f_t>
auto j_norm_sq(const std::vector<f_t>& u) -> f_t
{
  if (u.empty()) { return f_t(0); }

  f_t tail_sq = f_t(0);
  for (std::size_t j = 1; j < u.size(); ++j) {
    tail_sq += u[j] * u[j];
  }
  return u[0] * u[0] - tail_sq;
}

template <typename f_t>
auto tail_norm(const std::vector<f_t>& u) -> f_t
{
  f_t tail_sq = f_t(0);
  for (std::size_t j = 1; j < u.size(); ++j) {
    tail_sq += u[j] * u[j];
  }
  return std::sqrt(tail_sq);
}

template <typename f_t>
auto ref_apply_hinv_single(const std::vector<f_t>& z,
                           const std::vector<f_t>& w_bar,
                           f_t inv_eta,
                           f_t inv_1pw0) -> std::vector<f_t>
{
  std::vector<f_t> out(z.size(), f_t(0));
  if (z.empty()) { return out; }

  f_t zeta = f_t(0);
  for (std::size_t j = 1; j < z.size(); ++j) {
    zeta += w_bar[j] * z[j];
  }

  f_t coeff = -z[0] + zeta * inv_1pw0;
  out[0]    = (w_bar[0] * z[0] - zeta) * inv_eta;
  for (std::size_t j = 1; j < z.size(); ++j) {
    out[j] = (z[j] + coeff * w_bar[j]) * inv_eta;
  }
  return out;
}

template <typename f_t>
auto ref_apply_H_single(const std::vector<f_t>& z,
                        const std::vector<f_t>& w_bar,
                        f_t eta,
                        f_t inv_1pw0) -> std::vector<f_t>
{
  std::vector<f_t> out(z.size(), f_t(0));
  if (z.empty()) { return out; }

  f_t zeta = f_t(0);
  for (std::size_t j = 1; j < z.size(); ++j) {
    zeta += w_bar[j] * z[j];
  }

  f_t coeff = z[0] + zeta * inv_1pw0;
  out[0]    = (w_bar[0] * z[0] + zeta) * eta;
  for (std::size_t j = 1; j < z.size(); ++j) {
    out[j] = (z[j] + coeff * w_bar[j]) * eta;
  }
  return out;
}

template <typename f_t>
auto ref_apply_hinv2_single(const std::vector<f_t>& v, const std::vector<f_t>& w_bar, f_t inv_eta)
  -> std::vector<f_t>
{
  std::vector<f_t> out(v.size(), f_t(0));
  if (v.empty()) { return out; }

  f_t uTv = w_bar[0] * v[0];
  for (std::size_t j = 1; j < v.size(); ++j) {
    uTv -= w_bar[j] * v[j];
  }

  f_t ie_sq = inv_eta * inv_eta;
  out[0]    = ie_sq * (f_t(2) * w_bar[0] * uTv - v[0]);
  for (std::size_t j = 1; j < v.size(); ++j) {
    out[j] = ie_sq * (-f_t(2) * w_bar[j] * uTv + v[j]);
  }
  return out;
}

template <typename f_t>
struct nt_scaling_reference_t {
  f_t eta{};
  f_t inv_eta{};
  f_t inv_1pw0{};
  f_t rho{};
  std::vector<f_t> w_bar;
  std::vector<f_t> omega;
};

template <typename f_t>
auto ref_nt_scaling_single(const std::vector<f_t>& s, const std::vector<f_t>& lambda)
  -> nt_scaling_reference_t<f_t>
{
  EXPECT_EQ(s.size(), lambda.size());
  EXPECT_FALSE(s.empty());

  f_t s_j_norm_sq = j_norm_sq(s);
  f_t l_j_norm_sq = j_norm_sq(lambda);
  EXPECT_GT(s_j_norm_sq, f_t(0));
  EXPECT_GT(l_j_norm_sq, f_t(0));

  f_t s_j_norm     = std::sqrt(s_j_norm_sq);
  f_t l_j_norm     = std::sqrt(l_j_norm_sq);
  f_t inv_s_j_norm = f_t(1) / s_j_norm;
  f_t inv_l_j_norm = f_t(1) / l_j_norm;

  f_t dot_bar = (s[0] * lambda[0]) * inv_s_j_norm * inv_l_j_norm;
  for (std::size_t j = 1; j < s.size(); ++j) {
    dot_bar += (s[j] * lambda[j]) * inv_s_j_norm * inv_l_j_norm;
  }

  f_t gamma  = std::sqrt(std::max(f_t(0), (f_t(1) + dot_bar) * f_t(0.5)));
  f_t inv_2g = f_t(1) / (f_t(2) * gamma);

  nt_scaling_reference_t<f_t> ref{};
  ref.eta     = std::sqrt(s_j_norm / l_j_norm);
  ref.inv_eta = f_t(1) / ref.eta;
  ref.rho     = s_j_norm * l_j_norm;
  ref.w_bar.assign(s.size(), f_t(0));

  f_t w1_sq = f_t(0);
  for (std::size_t j = 1; j < s.size(); ++j) {
    ref.w_bar[j] = inv_2g * (s[j] * inv_s_j_norm - lambda[j] * inv_l_j_norm);
    w1_sq += ref.w_bar[j] * ref.w_bar[j];
  }

  // Match the kernel's numerical cleanup path for w_bar[0].
  ref.w_bar[0] = std::sqrt(f_t(1) + w1_sq);
  ref.inv_1pw0 = f_t(1) / (f_t(1) + ref.w_bar[0]);
  ref.omega    = ref_apply_hinv_single(s, ref.w_bar, ref.inv_eta, ref.inv_1pw0);
  return ref;
}

template <typename f_t>
auto ref_step_length_single(const std::vector<f_t>& u, const std::vector<f_t>& du, f_t alpha_max)
  -> f_t
{
  EXPECT_EQ(u.size(), du.size());
  EXPECT_FALSE(u.empty());

  f_t du1_sq = f_t(0);
  f_t u1du1  = f_t(0);
  f_t u1_sq  = f_t(0);
  for (std::size_t j = 1; j < u.size(); ++j) {
    du1_sq += du[j] * du[j];
    u1du1 += u[j] * du[j];
    u1_sq += u[j] * u[j];
  }

  f_t a    = du[0] * du[0] - du1_sq;
  f_t b    = u[0] * du[0] - u1du1;
  f_t c    = std::max(f_t(0), u[0] * u[0] - u1_sq);
  f_t disc = b * b - a * c;

  f_t alpha = alpha_max;
  if (du[0] < f_t(0)) { alpha = std::min(alpha, -u[0] / du[0]); }

  if ((a > f_t(0) && b > f_t(0)) || disc < f_t(0)) {
    // No positive root (parabola stays non-negative for alpha > 0).
  } else if (a < f_t(0)) {
    alpha = std::min(alpha, (b + std::sqrt(std::max(f_t(0), disc))) / (-a));
  } else if (a == f_t(0)) {
    if (b < f_t(0)) { alpha = std::min(alpha, c / (f_t(-2) * b)); }
  } else if (c == f_t(0)) {
    alpha = (a >= f_t(0)) ? alpha : f_t(0);
  } else if (b < f_t(0) && disc > f_t(0)) {
    alpha = std::min(alpha, (-b - std::sqrt(disc)) / a);
  }

  return alpha;
}

template <typename f_t>
auto ref_jordan_product_single(const std::vector<f_t>& a, const std::vector<f_t>& b)
  -> std::vector<f_t>
{
  EXPECT_EQ(a.size(), b.size());
  std::vector<f_t> out(a.size(), f_t(0));
  if (a.empty()) { return out; }

  f_t dot = f_t(0);
  for (std::size_t j = 0; j < a.size(); ++j) {
    dot += a[j] * b[j];
  }
  out[0] = dot;
  for (std::size_t j = 1; j < a.size(); ++j) {
    out[j] = a[0] * b[j] + b[0] * a[j];
  }
  return out;
}

template <typename f_t>
auto ref_inverse_jordan_product_single(const std::vector<f_t>& omega,
                                       const std::vector<f_t>& r,
                                       f_t rho_val) -> std::vector<f_t>
{
  EXPECT_EQ(omega.size(), r.size());
  std::vector<f_t> out(omega.size(), f_t(0));
  if (omega.empty()) { return out; }

  f_t nu = f_t(0);
  for (std::size_t j = 1; j < omega.size(); ++j) {
    nu += omega[j] * r[j];
  }

  f_t inv_rho = f_t(1) / rho_val;
  f_t omega_0 = omega[0];
  out[0]      = (omega_0 * r[0] - nu) * inv_rho;

  f_t c_omega = ((nu / omega_0) - r[0]) * inv_rho;
  f_t c_r     = f_t(1) / omega_0;
  for (std::size_t j = 1; j < omega.size(); ++j) {
    out[j] = c_omega * omega[j] + c_r * r[j];
  }
  return out;
}

template <typename f_t>
auto ref_fused_corrector_single(const std::vector<f_t>& dx_aff,
                                const std::vector<f_t>& omega,
                                const std::vector<f_t>& w_bar,
                                f_t inv_eta,
                                f_t inv_1pw0,
                                f_t rho_val,
                                f_t sigma_mu) -> std::vector<f_t>
{
  auto dx = ref_apply_hinv_single(dx_aff, w_bar, inv_eta, inv_1pw0);

  std::vector<f_t> dz(dx.size());
  for (std::size_t j = 0; j < dx.size(); ++j) {
    dz[j] = -omega[j] - dx[j];
  }

  auto r_K_1 = ref_jordan_product_single(omega, omega);
  auto r_K_2 = ref_jordan_product_single(dx, dz);

  std::vector<f_t> r_K(dx.size());
  for (std::size_t j = 0; j < dx.size(); ++j) {
    r_K[j] = r_K_1[j] + r_K_2[j];
  }
  r_K[0] -= sigma_mu;

  auto corr = ref_inverse_jordan_product_single(omega, r_K, rho_val);
  return ref_apply_hinv_single(corr, w_bar, inv_eta, inv_1pw0);
}

template <typename f_t>
auto ref_interior_shift_single(std::vector<f_t> u) -> std::vector<f_t>
{
  if (u.empty()) { return u; }

  f_t gap = tail_norm(u) - u[0];
  if (gap >= f_t(0)) { u[0] += f_t(1) + gap; }
  return u;
}

template <typename f_t>
auto make_patterned_cone(int q, f_t head, f_t scale) -> std::vector<f_t>
{
  std::vector<f_t> cone(q, f_t(0));
  cone[0] = head;
  for (int j = 1; j < q; ++j) {
    f_t sign = (j % 2 == 0) ? f_t(1) : f_t(-1);
    cone[j]  = sign * scale * static_cast<f_t>((j % 7) + 1);
  }
  return cone;
}

}  // namespace

class second_order_cone_test : public ::testing::Test {
 protected:
  using i_t                = int;
  using f_t                = double;
  static constexpr int dim = 256;

  raft::handle_t handle_;
  rmm::cuda_stream_view stream_ = handle_.get_stream();

  template <typename t_t>
  auto make_device_vector(const std::vector<t_t>& host) -> rmm::device_uvector<t_t>
  {
    rmm::device_uvector<t_t> device(host.size(), stream_);
    if (!host.empty()) { raft::copy(device.data(), host.data(), host.size(), stream_); }
    sync();
    return device;
  }

  template <typename t_t>
  auto copy_to_host(const rmm::device_uvector<t_t>& device) -> std::vector<t_t>
  {
    std::vector<t_t> host(device.size());
    if (!host.empty()) { raft::copy(host.data(), device.data(), host.size(), stream_); }
    sync();
    return host;
  }

  template <typename t_t>
  void copy_to_device(rmm::device_uvector<t_t>& device, const std::vector<t_t>& host)
  {
    ASSERT_EQ(device.size(), host.size());
    if (!host.empty()) { raft::copy(device.data(), host.data(), host.size(), stream_); }
    sync();
  }

  void sync() { RAFT_CUDA_TRY(cudaStreamSynchronize(stream_.value())); }

  template <typename t_t>
  void expect_vector_near(const std::vector<t_t>& actual,
                          const std::vector<t_t>& expected,
                          t_t atol,
                          t_t rtol,
                          const char* label)
  {
    ASSERT_EQ(actual.size(), expected.size()) << label << " size mismatch";
    for (std::size_t i = 0; i < actual.size(); ++i) {
      EXPECT_NEAR(actual[i], expected[i], atol + rtol * std::abs(expected[i]))
        << label << "[" << i << "]";
    }
  }

  void launch_apply_hinv(const rmm::device_uvector<f_t>& z,
                         rmm::device_uvector<f_t>& out,
                         const rmm::device_uvector<f_t>& w_bar,
                         const rmm::device_uvector<f_t>& inv_eta,
                         const rmm::device_uvector<f_t>& inv_1pw0,
                         const rmm::device_uvector<i_t>& cone_offsets,
                         i_t k)
  {
    apply_Hinv_kernel<i_t, f_t, dim><<<k, dim, 0, stream_>>>(
      z.data(), out.data(), w_bar.data(), inv_eta.data(), inv_1pw0.data(), cone_offsets.data(), k);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    sync();
  }

  void launch_step_length(const rmm::device_uvector<f_t>& s,
                          const rmm::device_uvector<f_t>& ds,
                          const rmm::device_uvector<f_t>& lambda,
                          const rmm::device_uvector<f_t>& dlambda,
                          rmm::device_uvector<f_t>& alpha,
                          const rmm::device_uvector<i_t>& cone_offsets,
                          i_t k,
                          f_t alpha_max)
  {
    step_length_kernel<i_t, f_t, dim><<<k, dim, 0, stream_>>>(s.data(),
                                                              ds.data(),
                                                              lambda.data(),
                                                              dlambda.data(),
                                                              alpha.data(),
                                                              cone_offsets.data(),
                                                              k,
                                                              alpha_max);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    sync();
  }

  void launch_interior_shift(rmm::device_uvector<f_t>& u,
                             const rmm::device_uvector<i_t>& cone_offsets,
                             i_t k)
  {
    interior_shift_kernel<i_t, f_t, dim><<<k, dim, 0, stream_>>>(u.data(), cone_offsets.data(), k);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    sync();
  }

  void launch_apply_hinv2(const rmm::device_uvector<f_t>& v,
                          rmm::device_uvector<f_t>& out,
                          const rmm::device_uvector<f_t>& w_bar,
                          const rmm::device_uvector<f_t>& inv_eta,
                          const rmm::device_uvector<i_t>& cone_offsets,
                          i_t k)
  {
    apply_Hinv2_kernel<i_t, f_t, dim><<<k, dim, 0, stream_>>>(
      v.data(), out.data(), w_bar.data(), inv_eta.data(), cone_offsets.data(), k);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    sync();
  }

  void launch_jordan_product(const rmm::device_uvector<f_t>& a,
                             const rmm::device_uvector<f_t>& b,
                             rmm::device_uvector<f_t>& out,
                             const rmm::device_uvector<i_t>& cone_offsets,
                             i_t k)
  {
    jordan_product_kernel<i_t, f_t, dim>
      <<<k, dim, 0, stream_>>>(a.data(), b.data(), out.data(), cone_offsets.data(), k);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    sync();
  }

  void launch_inverse_jordan_product(const rmm::device_uvector<f_t>& omega,
                                     const rmm::device_uvector<f_t>& r,
                                     const rmm::device_uvector<f_t>& rho,
                                     rmm::device_uvector<f_t>& out,
                                     const rmm::device_uvector<i_t>& cone_offsets,
                                     i_t k)
  {
    inverse_jordan_product_kernel<i_t, f_t, dim><<<k, dim, 0, stream_>>>(
      omega.data(), r.data(), rho.data(), out.data(), cone_offsets.data(), k);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    sync();
  }

  void launch_fused_corrector(const rmm::device_uvector<f_t>& dx_aff,
                              const rmm::device_uvector<f_t>& omega,
                              const rmm::device_uvector<f_t>& w_bar,
                              const rmm::device_uvector<f_t>& inv_eta,
                              const rmm::device_uvector<f_t>& inv_1pw0,
                              const rmm::device_uvector<f_t>& rho,
                              f_t sigma_mu,
                              rmm::device_uvector<f_t>& out,
                              const rmm::device_uvector<i_t>& cone_offsets,
                              i_t k)
  {
    fused_corrector_kernel<i_t, f_t, dim><<<k, dim, 0, stream_>>>(dx_aff.data(),
                                                                  omega.data(),
                                                                  w_bar.data(),
                                                                  inv_eta.data(),
                                                                  inv_1pw0.data(),
                                                                  rho.data(),
                                                                  sigma_mu,
                                                                  out.data(),
                                                                  cone_offsets.data(),
                                                                  k);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    sync();
  }
};

TEST_F(second_order_cone_test, cone_data_topology_and_bucket_partitioning)
{
  std::vector<i_t> dims{1, 32, 33, 2048, 2049};
  cone_data_t<i_t, f_t> cones(static_cast<i_t>(dims.size()), dims, stream_);

  auto expected_offsets = build_offsets(dims);
  auto actual_offsets   = copy_to_host(cones.cone_offsets);
  auto actual_dims      = copy_to_host(cones.cone_dims);
  auto small_ids        = copy_to_host(cones.small_cone_ids);
  auto medium_ids       = copy_to_host(cones.medium_cone_ids);
  auto large_ids        = copy_to_host(cones.large_cone_ids);

  EXPECT_EQ(cones.K, static_cast<i_t>(dims.size()));
  EXPECT_EQ(cones.m_c, expected_offsets.back());
  EXPECT_EQ(actual_offsets, expected_offsets);
  EXPECT_EQ(actual_dims, dims);
  EXPECT_EQ(small_ids, std::vector<i_t>({0, 1}));
  EXPECT_EQ(medium_ids, std::vector<i_t>({2, 3}));
  EXPECT_EQ(large_ids, std::vector<i_t>({4}));
}

TEST_F(second_order_cone_test, nt_scaling_matches_reference_for_small_cone)
{
  // Borrowed from the Clarabel regression input, but checked against our own
  // host-side NT formulas.
  std::vector<std::vector<f_t>> s_cones{{1.5, 0.3, 0.4}};
  std::vector<std::vector<f_t>> lambda_cones{{2.0, 0.5, 0.5}};
  std::vector<i_t> dims{3};

  cone_data_t<i_t, f_t> cones(1, dims, stream_);
  copy_to_device(cones.s, pack_cones(s_cones));
  copy_to_device(cones.lambda, pack_cones(lambda_cones));

  launch_nt_scaling(cones, stream_);

  auto eta      = copy_to_host(cones.eta);
  auto inv_eta  = copy_to_host(cones.inv_eta);
  auto inv_1pw0 = copy_to_host(cones.inv_1pw0);
  auto rho      = copy_to_host(cones.rho);
  auto w_bar    = copy_to_host(cones.w_bar);
  auto omega    = copy_to_host(cones.omega);

  auto ref = ref_nt_scaling_single(s_cones[0], lambda_cones[0]);

  EXPECT_NEAR(eta[0], ref.eta, 1e-12);
  EXPECT_NEAR(inv_eta[0], ref.inv_eta, 1e-12);
  EXPECT_NEAR(inv_1pw0[0], ref.inv_1pw0, 1e-12);
  EXPECT_NEAR(rho[0], ref.rho, 1e-12);
  expect_vector_near(w_bar, ref.w_bar, 1e-12, 1e-10, "w_bar");
  expect_vector_near(omega, ref.omega, 1e-12, 1e-10, "omega");

  EXPECT_NEAR(j_norm_sq(w_bar), f_t(1), 1e-12);
  EXPECT_NEAR(j_norm_sq(omega), rho[0], 1e-12);

  auto omega_from_apply_hinv = ref_apply_hinv_single(s_cones[0], w_bar, inv_eta[0], inv_1pw0[0]);
  expect_vector_near(omega, omega_from_apply_hinv, 1e-12, 1e-10, "omega_consistency");
}

TEST_F(second_order_cone_test, nt_scaling_matches_reference_across_bucket_sizes)
{
  std::vector<std::vector<f_t>> s_cones{
    {2.0}, make_patterned_cone<f_t>(33, 4.0, 0.01), make_patterned_cone<f_t>(2049, 5.0, 0.001)};
  std::vector<std::vector<f_t>> lambda_cones{
    {0.5}, make_patterned_cone<f_t>(33, 3.0, 0.0075), make_patterned_cone<f_t>(2049, 4.0, 0.00075)};
  std::vector<i_t> dims{1, 33, 2049};
  auto offsets = build_offsets(dims);

  cone_data_t<i_t, f_t> cones(static_cast<i_t>(dims.size()), dims, stream_);
  copy_to_device(cones.s, pack_cones(s_cones));
  copy_to_device(cones.lambda, pack_cones(lambda_cones));

  launch_nt_scaling(cones, stream_);

  auto eta      = copy_to_host(cones.eta);
  auto inv_eta  = copy_to_host(cones.inv_eta);
  auto inv_1pw0 = copy_to_host(cones.inv_1pw0);
  auto rho      = copy_to_host(cones.rho);
  auto w_bar    = copy_to_host(cones.w_bar);
  auto omega    = copy_to_host(cones.omega);

  for (i_t cone = 0; cone < static_cast<i_t>(dims.size()); ++cone) {
    auto ref = ref_nt_scaling_single(s_cones[cone], lambda_cones[cone]);

    EXPECT_NEAR(eta[cone], ref.eta, 1e-10) << "cone " << cone;
    EXPECT_NEAR(inv_eta[cone], ref.inv_eta, 1e-10) << "cone " << cone;
    EXPECT_NEAR(inv_1pw0[cone], ref.inv_1pw0, 1e-10) << "cone " << cone;
    EXPECT_NEAR(rho[cone], ref.rho, 1e-10) << "cone " << cone;

    auto actual_w_bar = slice_cone(w_bar, offsets, cone);
    auto actual_omega = slice_cone(omega, offsets, cone);
    expect_vector_near(actual_w_bar, ref.w_bar, 1e-10, 1e-8, "w_bar");
    expect_vector_near(actual_omega, ref.omega, 1e-10, 1e-8, "omega");

    EXPECT_NEAR(j_norm_sq(actual_w_bar), f_t(1), 1e-10) << "cone " << cone;
    EXPECT_NEAR(j_norm_sq(actual_omega), rho[cone], 1e-10) << "cone " << cone;
  }
}

TEST_F(second_order_cone_test, nt_scaling_omega_equals_H_times_lambda)
{
  std::vector<std::vector<f_t>> s_cones{{5.0, 1.0, -1.0, 0.5, 0.3}};
  std::vector<std::vector<f_t>> lambda_cones{{4.0, 0.5, 1.0, -0.3, 0.2}};
  std::vector<i_t> dims{5};

  cone_data_t<i_t, f_t> cones(1, dims, stream_);
  copy_to_device(cones.s, pack_cones(s_cones));
  copy_to_device(cones.lambda, pack_cones(lambda_cones));

  launch_nt_scaling(cones, stream_);

  auto eta      = copy_to_host(cones.eta);
  auto inv_1pw0 = copy_to_host(cones.inv_1pw0);
  auto w_bar    = copy_to_host(cones.w_bar);
  auto omega    = copy_to_host(cones.omega);

  // NT symmetry: omega should equal both H^{-1}s and H*lambda.
  auto H_lambda = ref_apply_H_single(lambda_cones[0], w_bar, eta[0], inv_1pw0[0]);
  expect_vector_near(omega, H_lambda, 1e-10, 1e-8, "omega_vs_H_lambda");
}

TEST_F(second_order_cone_test, nt_scaling_near_boundary_is_stable)
{
  // s and lambda barely inside the cone: ||tail||^2 ≈ head^2.
  std::vector<std::vector<f_t>> s_cones{{1.00002, 0.6, 0.8, 1e-4, -2e-4}};
  std::vector<std::vector<f_t>> lambda_cones{{1.000015, 0.8, 0.6, -3e-5, 2e-5}};
  std::vector<i_t> dims{5};

  cone_data_t<i_t, f_t> cones(1, dims, stream_);
  copy_to_device(cones.s, pack_cones(s_cones));
  copy_to_device(cones.lambda, pack_cones(lambda_cones));

  launch_nt_scaling(cones, stream_);

  auto eta      = copy_to_host(cones.eta);
  auto inv_eta  = copy_to_host(cones.inv_eta);
  auto inv_1pw0 = copy_to_host(cones.inv_1pw0);
  auto w_bar    = copy_to_host(cones.w_bar);
  auto omega    = copy_to_host(cones.omega);

  EXPECT_NEAR(j_norm_sq(w_bar), f_t(1), 1e-8) << "w_bar J-norm not 1 near boundary";
  EXPECT_GT(w_bar[0], tail_norm(w_bar)) << "w_bar not interior near boundary";

  // Round-trip: H(omega) should equal s.
  auto H_omega = ref_apply_H_single(omega, w_bar, eta[0], inv_1pw0[0]);
  expect_vector_near(H_omega, pack_cones(s_cones), 1e-8, 1e-6, "H_omega_vs_s_near_boundary");

  // Symmetry: omega should also equal H*lambda.
  auto H_lambda = ref_apply_H_single(lambda_cones[0], w_bar, eta[0], inv_1pw0[0]);
  expect_vector_near(omega, H_lambda, 1e-8, 1e-6, "omega_vs_H_lambda_near_boundary");
}

TEST_F(second_order_cone_test, apply_hinv_matches_reference_for_packed_cones)
{
  std::vector<i_t> dims{1, 3, 5};
  auto offsets = build_offsets(dims);

  std::vector<std::vector<f_t>> z_cones{{3.0}, {2.0, -1.0, 0.5}, {1.0, 0.25, -0.75, 0.5, -0.125}};
  std::vector<std::vector<f_t>> w_bar_cones{
    {1.0}, {0.0, 0.15, -0.05}, {0.0, 0.10, -0.20, 0.05, 0.15}};
  std::vector<f_t> inv_eta_host{0.5, 1.25, 0.75};
  std::vector<f_t> inv_1pw0_host(inv_eta_host.size(), 0.0);

  for (std::size_t cone = 0; cone < w_bar_cones.size(); ++cone) {
    f_t w1_sq = f_t(0);
    for (std::size_t j = 1; j < w_bar_cones[cone].size(); ++j) {
      w1_sq += w_bar_cones[cone][j] * w_bar_cones[cone][j];
    }
    w_bar_cones[cone][0] = std::sqrt(f_t(1) + w1_sq);
    inv_1pw0_host[cone]  = f_t(1) / (f_t(1) + w_bar_cones[cone][0]);
  }

  auto z         = make_device_vector(pack_cones(z_cones));
  auto w_bar     = make_device_vector(pack_cones(w_bar_cones));
  auto inv_eta   = make_device_vector(inv_eta_host);
  auto inv_1pw0  = make_device_vector(inv_1pw0_host);
  auto d_offsets = make_device_vector(offsets);
  rmm::device_uvector<f_t> out(z.size(), stream_);

  launch_apply_hinv(z, out, w_bar, inv_eta, inv_1pw0, d_offsets, static_cast<i_t>(dims.size()));

  auto actual_out = copy_to_host(out);
  auto expected   = pack_cones(std::vector<std::vector<f_t>>{
    ref_apply_hinv_single(z_cones[0], w_bar_cones[0], inv_eta_host[0], inv_1pw0_host[0]),
    ref_apply_hinv_single(z_cones[1], w_bar_cones[1], inv_eta_host[1], inv_1pw0_host[1]),
    ref_apply_hinv_single(z_cones[2], w_bar_cones[2], inv_eta_host[2], inv_1pw0_host[2])});

  expect_vector_near(actual_out, expected, 1e-12, 1e-10, "apply_hinv");
}

TEST_F(second_order_cone_test, step_length_matches_reference_and_handles_q1)
{
  std::vector<i_t> dims{1, 3};
  auto offsets = build_offsets(dims);

  std::vector<std::vector<f_t>> s_cones{{2.0}, {5.0, 1.0, 1.0}};
  std::vector<std::vector<f_t>> ds_cones{{-3.0}, {-0.5, 0.1, 0.1}};
  std::vector<std::vector<f_t>> lambda_cones{{5.0}, {5.0, 1.0, 1.0}};
  std::vector<std::vector<f_t>> dlambda_cones{{1.0}, {-0.5, 0.1, 0.1}};
  f_t alpha_max = 10.0;

  auto s         = make_device_vector(pack_cones(s_cones));
  auto ds        = make_device_vector(pack_cones(ds_cones));
  auto lambda    = make_device_vector(pack_cones(lambda_cones));
  auto dlambda   = make_device_vector(pack_cones(dlambda_cones));
  auto d_offsets = make_device_vector(offsets);
  rmm::device_uvector<f_t> alpha(dims.size(), stream_);

  launch_step_length(
    s, ds, lambda, dlambda, alpha, d_offsets, static_cast<i_t>(dims.size()), alpha_max);

  auto actual_alpha = copy_to_host(alpha);
  std::vector<f_t> expected_alpha(dims.size(), alpha_max);
  for (std::size_t cone = 0; cone < dims.size(); ++cone) {
    expected_alpha[cone] =
      std::min(ref_step_length_single(s_cones[cone], ds_cones[cone], alpha_max),
               ref_step_length_single(lambda_cones[cone], dlambda_cones[cone], alpha_max));
  }

  expect_vector_near(actual_alpha, expected_alpha, 1e-12, 1e-10, "step_length");
  EXPECT_NEAR(actual_alpha[0], 2.0 / 3.0, 1e-12);
  EXPECT_NEAR(actual_alpha[1], 5.5903758157691508, 1e-10);
}

TEST_F(second_order_cone_test, step_length_matches_reference_for_large_cone)
{
  std::vector<i_t> dims{513};
  auto offsets = build_offsets(dims);

  std::vector<std::vector<f_t>> s_cones{{make_patterned_cone<f_t>(dims[0], 5.0, 0.01)}};
  std::vector<std::vector<f_t>> ds_cones{{make_patterned_cone<f_t>(dims[0], -0.25, 0.002)}};
  std::vector<std::vector<f_t>> lambda_cones{{make_patterned_cone<f_t>(dims[0], 6.0, 0.009)}};
  std::vector<std::vector<f_t>> dlambda_cones{{make_patterned_cone<f_t>(dims[0], -0.15, 0.0015)}};
  f_t alpha_max = 20.0;

  auto s         = make_device_vector(pack_cones(s_cones));
  auto ds        = make_device_vector(pack_cones(ds_cones));
  auto lambda    = make_device_vector(pack_cones(lambda_cones));
  auto dlambda   = make_device_vector(pack_cones(dlambda_cones));
  auto d_offsets = make_device_vector(offsets);
  rmm::device_uvector<f_t> alpha(dims.size(), stream_);

  launch_step_length(
    s, ds, lambda, dlambda, alpha, d_offsets, static_cast<i_t>(dims.size()), alpha_max);

  auto actual_alpha = copy_to_host(alpha);
  std::vector<f_t> expected_alpha(dims.size(), alpha_max);
  for (std::size_t cone = 0; cone < dims.size(); ++cone) {
    expected_alpha[cone] =
      std::min(ref_step_length_single(s_cones[cone], ds_cones[cone], alpha_max),
               ref_step_length_single(lambda_cones[cone], dlambda_cones[cone], alpha_max));
  }

  expect_vector_near(actual_alpha, expected_alpha, 1e-12, 1e-10, "step_length_large");
  EXPECT_GT(actual_alpha[0], 0.0);
  EXPECT_LT(actual_alpha[0], alpha_max);
}

TEST_F(second_order_cone_test, step_length_boundary_c_zero_matches_clarabel_branch)
{
  std::vector<i_t> dims{3};
  auto offsets = build_offsets(dims);

  // Boundary point: c = u^T J u = 1^2 - 1^2 - 0^2 = 0.
  // Direction: a = du^T J du = 1^2 - 1^2 - 1^2 = -1 < 0.
  // Clarabel's c == 0 branch returns 0 in this case because the direction
  // leaves the cone immediately.
  std::vector<std::vector<f_t>> s_cones{{1.0, 1.0, 0.0}};
  std::vector<std::vector<f_t>> ds_cones{{1.0, 1.0, 1.0}};
  std::vector<std::vector<f_t>> lambda_cones{{1.0, 1.0, 0.0}};
  std::vector<std::vector<f_t>> dlambda_cones{{1.0, 1.0, 1.0}};
  f_t alpha_max = 10.0;

  auto s         = make_device_vector(pack_cones(s_cones));
  auto ds        = make_device_vector(pack_cones(ds_cones));
  auto lambda    = make_device_vector(pack_cones(lambda_cones));
  auto dlambda   = make_device_vector(pack_cones(dlambda_cones));
  auto d_offsets = make_device_vector(offsets);
  rmm::device_uvector<f_t> alpha(dims.size(), stream_);

  launch_step_length(
    s, ds, lambda, dlambda, alpha, d_offsets, static_cast<i_t>(dims.size()), alpha_max);

  auto actual_alpha = copy_to_host(alpha);
  ASSERT_EQ(actual_alpha.size(), 1);
  EXPECT_EQ(actual_alpha[0], 0.0);
}

TEST_F(second_order_cone_test, step_length_degenerate_a_zero)
{
  std::vector<i_t> dims{2};
  auto offsets = build_offsets(dims);

  // u=(2,0), du=(-1,1): a = du_0^2 - du_1^2 = 1 - 1 = 0 (degenerate quadratic).
  // Linear constraint: alpha <= 2. Degenerate branch: alpha = c/(-2b) = 4/2 = 2.
  // But the linear constraint also gives alpha <= 2, so result is min(2, 2) = 2...
  // Actually b = u0*du0 - u1*du1 = 2*(-1) - 0 = -2, c = u0^2 - u1^2 = 4.
  // Degenerate: alpha = c/(-2b) = 4/4 = 1. And linear: alpha <= -u0/du0 = 2.
  // So alpha = 1.
  std::vector<std::vector<f_t>> s_cones{{2.0, 0.0}};
  std::vector<std::vector<f_t>> ds_cones{{-1.0, 1.0}};
  std::vector<std::vector<f_t>> lambda_cones{{5.0, 0.0}};
  std::vector<std::vector<f_t>> dlambda_cones{{0.0, 0.0}};

  auto s         = make_device_vector(pack_cones(s_cones));
  auto ds        = make_device_vector(pack_cones(ds_cones));
  auto lambda    = make_device_vector(pack_cones(lambda_cones));
  auto dlambda   = make_device_vector(pack_cones(dlambda_cones));
  auto d_offsets = make_device_vector(offsets);
  rmm::device_uvector<f_t> alpha(1, stream_);

  launch_step_length(s, ds, lambda, dlambda, alpha, d_offsets, 1, 10.0);

  auto actual = copy_to_host(alpha);
  EXPECT_NEAR(actual[0], 1.0, 1e-14);
}

TEST_F(second_order_cone_test, step_length_safe_direction_returns_alpha_max)
{
  std::vector<i_t> dims{3};
  auto offsets = build_offsets(dims);

  // Interior point with direction along the identity element — stays in cone forever.
  std::vector<std::vector<f_t>> s_cones{{10.0, 0.0, 0.0}};
  std::vector<std::vector<f_t>> ds_cones{{1.0, 0.0, 0.0}};
  std::vector<std::vector<f_t>> lambda_cones{{10.0, 0.0, 0.0}};
  std::vector<std::vector<f_t>> dlambda_cones{{0.0, 0.1, 0.0}};

  auto s         = make_device_vector(pack_cones(s_cones));
  auto ds        = make_device_vector(pack_cones(ds_cones));
  auto lambda    = make_device_vector(pack_cones(lambda_cones));
  auto dlambda   = make_device_vector(pack_cones(dlambda_cones));
  auto d_offsets = make_device_vector(offsets);
  rmm::device_uvector<f_t> alpha(1, stream_);

  launch_step_length(s, ds, lambda, dlambda, alpha, d_offsets, 1, 1.0);

  auto actual = copy_to_host(alpha);
  EXPECT_DOUBLE_EQ(actual[0], 1.0);
}

TEST_F(second_order_cone_test, step_length_boundary_tightness)
{
  std::vector<i_t> dims{5};
  auto offsets = build_offsets(dims);

  std::vector<std::vector<f_t>> s_cones{{4.0, 1.0, -1.0, 0.5, 0.3}};
  std::vector<std::vector<f_t>> ds_cones{{-2.0, 1.0, 0.5, -0.3, 0.1}};
  std::vector<std::vector<f_t>> lambda_cones{{5.0, 0.5, 1.0, -0.3, 0.2}};
  std::vector<std::vector<f_t>> dlambda_cones{{-1.0, 2.0, 1.0, -0.5, 0.4}};

  auto s         = make_device_vector(pack_cones(s_cones));
  auto ds        = make_device_vector(pack_cones(ds_cones));
  auto lambda    = make_device_vector(pack_cones(lambda_cones));
  auto dlambda   = make_device_vector(pack_cones(dlambda_cones));
  auto d_offsets = make_device_vector(offsets);
  rmm::device_uvector<f_t> alpha(1, stream_);

  launch_step_length(s, ds, lambda, dlambda, alpha, d_offsets, 1, 100.0);

  auto a = copy_to_host(alpha)[0];
  ASSERT_GT(a, 0.0);

  // At alpha, at least one of (s, lambda) should be on the cone boundary.
  auto s_bnd = s_cones[0];
  auto l_bnd = lambda_cones[0];
  for (std::size_t j = 0; j < s_bnd.size(); ++j) {
    s_bnd[j] += a * ds_cones[0][j];
    l_bnd[j] += a * dlambda_cones[0][j];
  }
  f_t res_s = j_norm_sq(s_bnd);
  f_t res_l = j_norm_sq(l_bnd);
  EXPECT_GE(res_s, -1e-10) << "s left the cone";
  EXPECT_GE(res_l, -1e-10) << "lambda left the cone";
  EXPECT_NEAR(std::min(res_s, res_l), 0.0, 1e-10) << "neither hit the boundary";

  // At (1 − ε) α, both should be strictly interior.
  f_t a_int  = a * (1.0 - 1e-8);
  auto s_int = s_cones[0];
  auto l_int = lambda_cones[0];
  for (std::size_t j = 0; j < s_int.size(); ++j) {
    s_int[j] += a_int * ds_cones[0][j];
    l_int[j] += a_int * dlambda_cones[0][j];
  }
  EXPECT_GT(j_norm_sq(s_int), 0.0) << "s not interior at (1-eps)*alpha";
  EXPECT_GT(j_norm_sq(l_int), 0.0) << "lambda not interior at (1-eps)*alpha";
}

TEST_F(second_order_cone_test, interior_shift_matches_reference_and_preserves_tail)
{
  std::vector<i_t> dims{1, 3, 4};
  auto offsets = build_offsets(dims);

  std::vector<std::vector<f_t>> cones{{-0.25}, {2.0, 0.3, 0.4}, {0.5, 0.6, 0.8, 0.0}};
  auto packed   = pack_cones(cones);
  auto expected = pack_cones(std::vector<std::vector<f_t>>{ref_interior_shift_single(cones[0]),
                                                           ref_interior_shift_single(cones[1]),
                                                           ref_interior_shift_single(cones[2])});

  auto u         = make_device_vector(packed);
  auto d_offsets = make_device_vector(offsets);
  launch_interior_shift(u, d_offsets, static_cast<i_t>(dims.size()));

  auto actual = copy_to_host(u);
  expect_vector_near(actual, expected, 1e-12, 1e-10, "interior_shift");

  for (std::size_t cone = 0; cone < dims.size(); ++cone) {
    auto shifted = slice_cone(actual, offsets, static_cast<i_t>(cone));
    EXPECT_GT(shifted[0], tail_norm(shifted));
    for (std::size_t j = 1; j < shifted.size(); ++j) {
      EXPECT_EQ(shifted[j], cones[cone][j]) << "cone " << cone << " tail " << j;
    }
  }
}

TEST_F(second_order_cone_test, apply_hinv2_matches_reference_for_packed_cones)
{
  std::vector<i_t> dims{1, 3, 5};
  auto offsets = build_offsets(dims);

  std::vector<std::vector<f_t>> v_cones{{3.0}, {2.0, -1.0, 0.5}, {1.0, 0.25, -0.75, 0.5, -0.125}};
  std::vector<std::vector<f_t>> w_bar_cones{
    {1.0}, {0.0, 0.15, -0.05}, {0.0, 0.10, -0.20, 0.05, 0.15}};
  std::vector<f_t> inv_eta_host{0.5, 1.25, 0.75};

  for (std::size_t cone = 0; cone < w_bar_cones.size(); ++cone) {
    f_t w1_sq = f_t(0);
    for (std::size_t j = 1; j < w_bar_cones[cone].size(); ++j) {
      w1_sq += w_bar_cones[cone][j] * w_bar_cones[cone][j];
    }
    w_bar_cones[cone][0] = std::sqrt(f_t(1) + w1_sq);
  }

  auto v         = make_device_vector(pack_cones(v_cones));
  auto w_bar     = make_device_vector(pack_cones(w_bar_cones));
  auto inv_eta   = make_device_vector(inv_eta_host);
  auto d_offsets = make_device_vector(offsets);
  rmm::device_uvector<f_t> out(v.size(), stream_);

  launch_apply_hinv2(v, out, w_bar, inv_eta, d_offsets, static_cast<i_t>(dims.size()));

  auto actual   = copy_to_host(out);
  auto expected = pack_cones(std::vector<std::vector<f_t>>{
    ref_apply_hinv2_single(v_cones[0], w_bar_cones[0], inv_eta_host[0]),
    ref_apply_hinv2_single(v_cones[1], w_bar_cones[1], inv_eta_host[1]),
    ref_apply_hinv2_single(v_cones[2], w_bar_cones[2], inv_eta_host[2])});

  expect_vector_near(actual, expected, 1e-12, 1e-10, "apply_hinv2");
}

TEST_F(second_order_cone_test, apply_hinv2_equals_double_hinv_with_nt_scaling)
{
  std::vector<std::vector<f_t>> s_cones{{2.0, 0.5, 0.25}, {3.0, 0.25, -0.5, 0.75, -0.25}};
  std::vector<std::vector<f_t>> lambda_cones{{1.5, -0.25, 0.1}, {2.5, -0.1, 0.3, -0.2, 0.15}};
  std::vector<i_t> dims{3, 5};
  auto offsets = build_offsets(dims);

  cone_data_t<i_t, f_t> cones(static_cast<i_t>(dims.size()), dims, stream_);
  copy_to_device(cones.s, pack_cones(s_cones));
  copy_to_device(cones.lambda, pack_cones(lambda_cones));

  launch_nt_scaling(cones, stream_);

  std::vector<std::vector<f_t>> v_cones{{1.0, -0.3, 0.2}, {0.5, 0.1, -0.15, 0.25, -0.1}};
  auto d_v       = make_device_vector(pack_cones(v_cones));
  auto d_offsets = make_device_vector(offsets);

  // H^{-2} v  (single kernel)
  rmm::device_uvector<f_t> d_hinv2(cones.omega.size(), stream_);
  launch_apply_hinv2(
    d_v, d_hinv2, cones.w_bar, cones.inv_eta, d_offsets, static_cast<i_t>(dims.size()));

  // H^{-1}(H^{-1} v)  (two passes)
  rmm::device_uvector<f_t> d_tmp(cones.omega.size(), stream_);
  rmm::device_uvector<f_t> d_double(cones.omega.size(), stream_);
  launch_apply_hinv(d_v,
                    d_tmp,
                    cones.w_bar,
                    cones.inv_eta,
                    cones.inv_1pw0,
                    d_offsets,
                    static_cast<i_t>(dims.size()));
  launch_apply_hinv(d_tmp,
                    d_double,
                    cones.w_bar,
                    cones.inv_eta,
                    cones.inv_1pw0,
                    d_offsets,
                    static_cast<i_t>(dims.size()));

  auto hinv2_actual  = copy_to_host(d_hinv2);
  auto double_actual = copy_to_host(d_double);
  expect_vector_near(hinv2_actual, double_actual, 1e-10, 1e-8, "hinv2_vs_double_hinv");
}

TEST_F(second_order_cone_test, apply_hinv2_strided_loop_for_large_cone)
{
  std::vector<i_t> dims{513};
  auto offsets = build_offsets(dims);

  auto s_cone      = make_patterned_cone<f_t>(dims[0], 5.0, 0.005);
  auto lambda_cone = make_patterned_cone<f_t>(dims[0], 4.0, 0.004);

  cone_data_t<i_t, f_t> cones(1, dims, stream_);
  copy_to_device(cones.s, s_cone);
  copy_to_device(cones.lambda, lambda_cone);

  launch_nt_scaling(cones, stream_);

  auto v_cone    = make_patterned_cone<f_t>(dims[0], 3.0, 0.006);
  auto d_v       = make_device_vector(v_cone);
  auto d_offsets = make_device_vector(offsets);

  // Direct H^{-2} apply
  rmm::device_uvector<f_t> d_hinv2(cones.omega.size(), stream_);
  launch_apply_hinv2(d_v, d_hinv2, cones.w_bar, cones.inv_eta, d_offsets, 1);

  // Reference: two H^{-1} passes
  rmm::device_uvector<f_t> d_tmp(cones.omega.size(), stream_);
  rmm::device_uvector<f_t> d_double(cones.omega.size(), stream_);
  launch_apply_hinv(d_v, d_tmp, cones.w_bar, cones.inv_eta, cones.inv_1pw0, d_offsets, 1);
  launch_apply_hinv(d_tmp, d_double, cones.w_bar, cones.inv_eta, cones.inv_1pw0, d_offsets, 1);

  auto hinv2_actual  = copy_to_host(d_hinv2);
  auto double_actual = copy_to_host(d_double);
  expect_vector_near(hinv2_actual, double_actual, 1e-8, 1e-6, "hinv2_large");

  // Also check against CPU reference
  auto w_bar_host   = copy_to_host(cones.w_bar);
  auto inv_eta_host = copy_to_host(cones.inv_eta);
  auto ref          = ref_apply_hinv2_single(v_cone, w_bar_host, inv_eta_host[0]);
  expect_vector_near(hinv2_actual, ref, 1e-8, 1e-6, "hinv2_large_ref");
}

TEST_F(second_order_cone_test, jordan_product_matches_reference_for_packed_cones)
{
  std::vector<i_t> dims{1, 3, 4};
  auto offsets = build_offsets(dims);

  std::vector<std::vector<f_t>> a_cones{{2.0}, {2.0, 1.0, -0.5}, {3.0, 0.25, -0.75, 0.5}};
  std::vector<std::vector<f_t>> b_cones{{4.0}, {1.5, -0.5, 0.25}, {2.0, -0.25, 0.5, 1.0}};

  auto a         = make_device_vector(pack_cones(a_cones));
  auto b         = make_device_vector(pack_cones(b_cones));
  auto d_offsets = make_device_vector(offsets);
  rmm::device_uvector<f_t> out(a.size(), stream_);

  launch_jordan_product(a, b, out, d_offsets, static_cast<i_t>(dims.size()));

  auto actual = copy_to_host(out);
  auto expected =
    pack_cones(std::vector<std::vector<f_t>>{ref_jordan_product_single(a_cones[0], b_cones[0]),
                                             ref_jordan_product_single(a_cones[1], b_cones[1]),
                                             ref_jordan_product_single(a_cones[2], b_cones[2])});

  expect_vector_near(actual, expected, 1e-12, 1e-10, "jordan_product");
}

TEST_F(second_order_cone_test, inverse_jordan_product_matches_reference_and_identity)
{
  std::vector<i_t> dims{1, 3, 5};
  auto offsets = build_offsets(dims);

  std::vector<std::vector<f_t>> omega_cones{
    {2.0}, {2.0, 0.5, 0.25}, {3.0, 0.25, -0.5, 0.75, -0.25}};
  std::vector<std::vector<f_t>> r_cones{{4.0}, {1.0, -0.25, 0.5}, {2.0, 0.5, -0.25, 0.25, 0.75}};

  std::vector<f_t> rho_host;
  for (const auto& w : omega_cones) {
    rho_host.push_back(j_norm_sq(w));
  }

  auto d_omega   = make_device_vector(pack_cones(omega_cones));
  auto d_r       = make_device_vector(pack_cones(r_cones));
  auto d_rho     = make_device_vector(rho_host);
  auto d_offsets = make_device_vector(offsets);
  rmm::device_uvector<f_t> d_out(d_omega.size(), stream_);

  launch_inverse_jordan_product(
    d_omega, d_r, d_rho, d_out, d_offsets, static_cast<i_t>(dims.size()));

  auto actual   = copy_to_host(d_out);
  auto expected = pack_cones(std::vector<std::vector<f_t>>{
    ref_inverse_jordan_product_single(omega_cones[0], r_cones[0], rho_host[0]),
    ref_inverse_jordan_product_single(omega_cones[1], r_cones[1], rho_host[1]),
    ref_inverse_jordan_product_single(omega_cones[2], r_cones[2], rho_host[2])});

  expect_vector_near(actual, expected, 1e-12, 1e-10, "inverse_jordan_product");

  // Identity check: omega circ (omega \ r) = r
  auto d_inv = make_device_vector(actual);
  rmm::device_uvector<f_t> d_roundtrip(d_omega.size(), stream_);
  launch_jordan_product(d_omega, d_inv, d_roundtrip, d_offsets, static_cast<i_t>(dims.size()));

  auto roundtrip = copy_to_host(d_roundtrip);
  auto r_packed  = pack_cones(r_cones);
  expect_vector_near(roundtrip, r_packed, 1e-10, 1e-8, "inverse_identity");
}

TEST_F(second_order_cone_test, jordan_and_inverse_jordan_strided_loop_for_large_cone)
{
  std::vector<i_t> dims{513};
  auto offsets = build_offsets(dims);

  auto a_cone     = make_patterned_cone<f_t>(dims[0], 5.0, 0.005);
  auto b_cone     = make_patterned_cone<f_t>(dims[0], 4.0, 0.004);
  auto omega_cone = make_patterned_cone<f_t>(dims[0], 6.0, 0.003);
  f_t rho_val     = j_norm_sq(omega_cone);
  ASSERT_GT(rho_val, 0.0);

  std::vector<std::vector<f_t>> a_cones{a_cone};
  std::vector<std::vector<f_t>> b_cones{b_cone};
  std::vector<std::vector<f_t>> omega_cones{omega_cone};

  auto d_a       = make_device_vector(pack_cones(a_cones));
  auto d_b       = make_device_vector(pack_cones(b_cones));
  auto d_omega   = make_device_vector(pack_cones(omega_cones));
  auto d_rho     = make_device_vector(std::vector<f_t>{rho_val});
  auto d_offsets = make_device_vector(offsets);

  // Jordan product: strided path
  rmm::device_uvector<f_t> d_jp(d_a.size(), stream_);
  launch_jordan_product(d_a, d_b, d_jp, d_offsets, 1);
  auto jp_actual   = copy_to_host(d_jp);
  auto jp_expected = ref_jordan_product_single(a_cone, b_cone);
  expect_vector_near(jp_actual, jp_expected, 1e-10, 1e-8, "jordan_large");

  // Inverse Jordan product: strided path + identity
  auto r_cone = make_patterned_cone<f_t>(dims[0], 3.0, 0.006);
  std::vector<std::vector<f_t>> r_cones{r_cone};

  auto d_r = make_device_vector(pack_cones(r_cones));
  rmm::device_uvector<f_t> d_inv(d_omega.size(), stream_);
  launch_inverse_jordan_product(d_omega, d_r, d_rho, d_inv, d_offsets, 1);

  auto inv_actual   = copy_to_host(d_inv);
  auto inv_expected = ref_inverse_jordan_product_single(omega_cone, r_cone, rho_val);
  expect_vector_near(inv_actual, inv_expected, 1e-10, 1e-8, "inv_jordan_large");

  // Round-trip identity on the large cone
  auto d_inv_vec = make_device_vector(inv_actual);
  rmm::device_uvector<f_t> d_rt(d_omega.size(), stream_);
  launch_jordan_product(d_omega, d_inv_vec, d_rt, d_offsets, 1);
  auto rt_actual = copy_to_host(d_rt);
  expect_vector_near(rt_actual, r_cone, 1e-8, 1e-6, "identity_large");
}

TEST_F(second_order_cone_test, inverse_jordan_product_with_nt_scaling_rho)
{
  std::vector<std::vector<f_t>> s_cones{{2.0, 0.5, 0.25}, {3.0, 0.25, -0.5, 0.75, -0.25}};
  std::vector<std::vector<f_t>> lambda_cones{{1.5, -0.25, 0.1}, {2.5, -0.1, 0.3, -0.2, 0.15}};
  std::vector<i_t> dims{3, 5};
  auto offsets = build_offsets(dims);

  cone_data_t<i_t, f_t> cones(static_cast<i_t>(dims.size()), dims, stream_);
  copy_to_device(cones.s, pack_cones(s_cones));
  copy_to_device(cones.lambda, pack_cones(lambda_cones));

  launch_nt_scaling(cones, stream_);

  auto omega_host = copy_to_host(cones.omega);
  auto rho_host   = copy_to_host(cones.rho);

  // Build an arbitrary r vector, run inverse Jordan with NT-produced rho/omega
  std::vector<std::vector<f_t>> r_cones{{1.0, -0.3, 0.2}, {0.5, 0.1, -0.15, 0.25, -0.1}};
  auto d_r       = make_device_vector(pack_cones(r_cones));
  auto d_offsets = make_device_vector(offsets);
  rmm::device_uvector<f_t> d_out(cones.omega.size(), stream_);

  launch_inverse_jordan_product(
    cones.omega, d_r, cones.rho, d_out, d_offsets, static_cast<i_t>(dims.size()));

  auto inv_actual = copy_to_host(d_out);

  // Verify host reference matches using NT-produced values
  for (i_t c = 0; c < static_cast<i_t>(dims.size()); ++c) {
    auto omega_c = slice_cone(omega_host, offsets, c);
    auto r_c     = r_cones[c];
    auto ref     = ref_inverse_jordan_product_single(omega_c, r_c, rho_host[c]);
    auto actual  = slice_cone(inv_actual, offsets, c);
    expect_vector_near(actual, ref, 1e-10, 1e-8, "nt_rho_inv_jordan");
  }

  // Round-trip identity with NT-produced omega
  rmm::device_uvector<f_t> d_rt(cones.omega.size(), stream_);
  auto d_inv = make_device_vector(inv_actual);
  launch_jordan_product(cones.omega, d_inv, d_rt, d_offsets, static_cast<i_t>(dims.size()));
  auto rt_actual = copy_to_host(d_rt);
  auto r_packed  = pack_cones(r_cones);
  expect_vector_near(rt_actual, r_packed, 1e-8, 1e-6, "nt_identity");
}

TEST_F(second_order_cone_test, fused_corrector_matches_reference_with_nt_scaling)
{
  std::vector<std::vector<f_t>> s_cones{{2.0, 0.5, 0.25}, {3.0, 0.25, -0.5, 0.75, -0.25}};
  std::vector<std::vector<f_t>> lambda_cones{{1.5, -0.25, 0.1}, {2.5, -0.1, 0.3, -0.2, 0.15}};
  std::vector<i_t> dims{3, 5};
  auto offsets = build_offsets(dims);

  cone_data_t<i_t, f_t> cones(static_cast<i_t>(dims.size()), dims, stream_);
  copy_to_device(cones.s, pack_cones(s_cones));
  copy_to_device(cones.lambda, pack_cones(lambda_cones));

  launch_nt_scaling(cones, stream_);

  std::vector<std::vector<f_t>> dx_aff_cones{{0.3, -0.1, 0.2}, {-0.5, 0.2, 0.1, -0.3, 0.15}};
  f_t sigma_mu = 0.1;

  auto d_dx_aff  = make_device_vector(pack_cones(dx_aff_cones));
  auto d_offsets = make_device_vector(offsets);
  rmm::device_uvector<f_t> d_out(cones.omega.size(), stream_);

  launch_fused_corrector(d_dx_aff,
                         cones.omega,
                         cones.w_bar,
                         cones.inv_eta,
                         cones.inv_1pw0,
                         cones.rho,
                         sigma_mu,
                         d_out,
                         d_offsets,
                         static_cast<i_t>(dims.size()));

  auto actual     = copy_to_host(d_out);
  auto omega_host = copy_to_host(cones.omega);
  auto w_bar_host = copy_to_host(cones.w_bar);
  auto inv_eta_h  = copy_to_host(cones.inv_eta);
  auto inv_1pw0_h = copy_to_host(cones.inv_1pw0);
  auto rho_h      = copy_to_host(cones.rho);

  for (i_t c = 0; c < static_cast<i_t>(dims.size()); ++c) {
    auto ref = ref_fused_corrector_single(dx_aff_cones[c],
                                          slice_cone(omega_host, offsets, c),
                                          slice_cone(w_bar_host, offsets, c),
                                          inv_eta_h[c],
                                          inv_1pw0_h[c],
                                          rho_h[c],
                                          sigma_mu);
    auto act = slice_cone(actual, offsets, c);
    expect_vector_near(act, ref, 1e-10, 1e-8, "fused_corrector");
  }
}

TEST_F(second_order_cone_test, fused_corrector_strided_loop_for_large_cone)
{
  std::vector<i_t> dims{513};
  auto offsets = build_offsets(dims);

  auto s_cone      = make_patterned_cone<f_t>(dims[0], 5.0, 0.005);
  auto lambda_cone = make_patterned_cone<f_t>(dims[0], 4.0, 0.004);

  cone_data_t<i_t, f_t> cones(1, dims, stream_);
  copy_to_device(cones.s, s_cone);
  copy_to_device(cones.lambda, lambda_cone);

  launch_nt_scaling(cones, stream_);

  auto dx_aff_cone = make_patterned_cone<f_t>(dims[0], 0.5, 0.003);
  f_t sigma_mu     = 0.25;

  auto d_dx_aff  = make_device_vector(dx_aff_cone);
  auto d_offsets = make_device_vector(offsets);
  rmm::device_uvector<f_t> d_out(cones.omega.size(), stream_);

  launch_fused_corrector(d_dx_aff,
                         cones.omega,
                         cones.w_bar,
                         cones.inv_eta,
                         cones.inv_1pw0,
                         cones.rho,
                         sigma_mu,
                         d_out,
                         d_offsets,
                         1);

  auto actual     = copy_to_host(d_out);
  auto omega_host = copy_to_host(cones.omega);
  auto w_bar_host = copy_to_host(cones.w_bar);
  auto inv_eta_h  = copy_to_host(cones.inv_eta);
  auto inv_1pw0_h = copy_to_host(cones.inv_1pw0);
  auto rho_h      = copy_to_host(cones.rho);

  auto ref = ref_fused_corrector_single(
    dx_aff_cone, omega_host, w_bar_host, inv_eta_h[0], inv_1pw0_h[0], rho_h[0], sigma_mu);
  expect_vector_near(actual, ref, 1e-8, 1e-6, "fused_corrector_large");
}

}  // namespace cuopt::linear_programming::dual_simplex::test
