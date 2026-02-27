#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include <array>
#include <random>
#include <map>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/cfg/env.h>

#include <boost/algorithm/string.hpp>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "cosmolike/basics.h"
#include "cosmolike/bias.h"
#include "cosmolike/baryons.h"
#include "cosmolike/cosmo2D.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo.h"
#include "cosmolike/radial_weights.h"
#include "cosmolike/pt_cfastpt.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"

#include <carma.h>
#include <armadillo>
#include "cosmolike/generic_interface.hpp"
#include "cosmolike/cosmo2D_wrapper.hpp"

// Some helper functions for debugging
static int include_HOD_GX = 0; // 0 or 1
static int include_RSD_GS = 0; // 0 or 1 
arma::Col<double> kernel_for_C_ss_tomo_limber(double a, arma::Col<double> params)
{
  /* params: [n1, n2, l, flag_EE]
  Return: coefficients for [Pk, ta, tt, mix, ta_dE1+ta_dE2, mixA+mixB, mixEE] + k (dimensionless)
   */
  if (!(a>0) || !(a<1)) {
    spdlog::debug("\x1b[90m{}\x1b[0m: a>0 and a<1 not true", "kernel_for_C_ss_tomo_limber");
    exit(1);
  }
  const int n1 = (int) params(0); // first source bin 
  const int n2 = (int) params(1); // second source bin 
  if (n1 < 0 || 
      n1 > redshift.shear_nbin - 1 || 
      n2 < 0 || 
      n2 > redshift.shear_nbin - 1)
  {
    spdlog::debug("\x1b[90m{}\x1b[0m: error in selecting bin number (ni, nj) = [{},{}]", "kernel_for_C_ss_tomo_limber", n1, n2);
    exit(1);
  }
  const double l = params(2);
  const int EE = (int) params(3);

  const double ell = l + 0.5;
  struct chis chidchi = chi_all(a);
  const double growfac_a = growfac(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double fK = f_K(chidchi.chi); // (Mpc/h)/(c/H0=100) (dimensionless)
  const double k = ell/fK; // (c/H0)/(Mpc/h)
  
  const double WK1 = W_kappa(a, fK, n1);
  const double WK2 = W_kappa(a, fK, n2);
  const double WS1 = W_source(a, n1, hoverh0);
  const double WS2 = W_source(a, n2, hoverh0);

  const double ell4 = ell*ell*ell*ell; // correction (1812.05995 eqs 74-79)
  const double ell_prefactor = l*(l - 1.)*(l + 1.)*(l + 2.)/ell4; 

  arma::Col ans = arma::zeros<arma::Col<double>>(8);
  ans(7) = k;
  switch(nuisance.IA_MODEL) 
  {
    case IA_MODEL_TATT:
    { 
      // JX: get IA coefficients at different redshift
      double IA_AX[2];
      IA_A1_Z1Z2(a, growfac_a, n1, n2, IA_AX);
      const double C11 = IA_AX[0];
      const double C12 = IA_AX[1];
      IA_A2_Z1Z2(a, growfac_a, n1, n2, IA_AX);
      const double C21 = IA_AX[0];
      const double C22 = IA_AX[1];
      IA_BTA_Z1Z2(a, growfac_a, n1, n2, IA_AX);
      const double bta1 = IA_AX[0];
      const double bta2 = IA_AX[1];

      if (EE == 1)
      {
        ans(0) = WK1*WK2 - WS1*WK2*C11 - WS2*WK1*C12 + WS1*WS2*C11*C12; // Pk
        ans(1) = WS1*WS2*C11*C12*bta1*bta2; // ta
        ans(2) = WS1*WS2*25.*C21*C22; // tt
        ans(4) = -WS1*WK2*C11*bta1 - WS2*WK1*C12*bta2 + WS1*WS2*C11*C12*(bta1+bta2); // ta_dE1+ta_dE2
        ans(5) = WS1*WK2*5.*C21 + WS2*WK1*5.*C22 - WS1*WS2*5.*(C11*C22 + C12*C21); // mixA+mixB
        ans(6) = -WS1*WS2*5.*(C11*bta1*C22+C12*bta2*C21); // mixEE
      }
      else  
      {
        ans(0) = 0.; // Pk
        ans(1) = WS1*WS2*C11*C12*bta1*bta2; // ta
        ans(2) = WS1*WS2*25.*C21*C22; // tt
        ans(3) = -WS1*WS2*5.*(C11*bta1*C22+C12*bta2*C21); // mix
      }
      break;
    }
    case IA_MODEL_NLA:
    {
      if (EE == 1) { 
        double IA_A1[2];
        IA_A1_Z1Z2(a, growfac_a, n1, n2, IA_A1);
        const double C11 = IA_A1[0];
        const double C12 = IA_A1[1];
        ans(0) = WK1*WK2 - WS1*WK2*C11 - WS2*WK1*C12 + WS1*WS2*C11*C12;
      }
      break;
    }
    default:
    {
      exit(1);
    }
  }
  return ans*(chidchi.dchida/(fK*fK))*ell_prefactor;
}

arma::Col<double> kernel_for_C_gs_tomo_limber(double a, arma::Col<double> params)
{
  /* params: [nl, ns, l, flag_nonlinear_bias]
  Return: coefficients for [Pk, 1loop, ta, tt, mix, ta_dE1+ta_dE2, mixA+mixB, mixEE] + k (dimensionless)
   */
  if (!(a>0) || !(a<1)) {
    spdlog::debug("\x1b[90m{}\x1b[0m: a>0 and a<1 not true", "kernel_for_C_gs_tomo_limber");
    exit(1);
  }
  const int nl = (int) params(0);
  const int ns = (int) params(1);
  if (nl < 0 || 
      nl > redshift.clustering_nbin - 1 || 
      ns < 0 || 
      ns > redshift.shear_nbin - 1) {
    spdlog::debug("\x1b[90m{}\x1b[0m: error in selecting bin number (nl, ns) = [{},{}]", "kernel_for_C_gs_tomo_limber", nl, ns);
    exit(1);
  }
  const double l = params(2);
  const int nonlinear_bias = params(3);
  
  const double growfac_a = growfac(a);
  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double ell = l + 0.5;
  const double fK = f_K(chidchi.chi);
  const double k = ell/fK;
  const double z = 1.0/a - 1.0;

  const double b1 = gb1(z, nl);
  const double bmag = gbmag(z, nl);

  const double WK = W_kappa(a, fK, ns);
  const double WGAL = W_gal(a, nl, hoverh0);
  const double WMAG = W_mag(a, fK, nl);
  const double WS   = W_source(a, ns, hoverh0);

  const double ell_prefactor = l*(l + 1.)/(ell*ell); // correction (1812.05995 eqs 74-79)
  const double tmp = (l - 1.)*l*(l + 1.)*(l + 2.);   // correction (1812.05995 eqs 74-79)
  const double ell_prefactor2 = (tmp > 0) ? sqrt(tmp)/(ell*ell) : 0.0;

  arma::Col<double> ans = arma::zeros<arma::Col<double>>(9);
  ans(8) = k;

  switch(nuisance.IA_MODEL)
  {
    case IA_MODEL_TATT:
    {
      if (include_HOD_GX == 1)
      {
        spdlog::debug("\x1b[90m{}\x1b[0m: HOD not implemented!", "kernel_for_C_gs_tomo_limber");
        exit(1);
      }

      double WRSD = 0.0;
      if (include_RSD_GS == 1)
      {
        const double chi_0 = f_K(ell/k);
        const double chi_1 = f_K((ell+1.)/k);
        const double a_0 = a_chi(chi_0);
        const double a_1 = a_chi(chi_1);
        WRSD = W_RSD(ell, a_0, a_1, nl);
      }

      const double C1ZS  = IA_A1_Z1(a, growfac_a, ns);
      const double btazs = IA_BTA_Z1(a, growfac_a, ns);
      const double C2ZS  = IA_A2_Z1(a, growfac_a, ns);

      // TODO: IS THIS CONSISTENT (WRSD, ONELOOP AND IA CROSS TERMS)?
      ans(0) = WK*(WGAL*b1+WMAG*ell_prefactor*bmag+WRSD)-WS*(WGAL*b1+WMAG*ell_prefactor*bmag)*C1ZS; // Pk
      ans(1) = WK*WGAL; // 1loop
      ans(5) = -WS*(WGAL*b1+WMAG*ell_prefactor*bmag)*C1ZS*btazs; // ta_dE1+ta_dE2
      ans(6) = WS*(WGAL*b1+WMAG*ell_prefactor*bmag)*5.*C2ZS; // mixA+mixB
      break;
    }
    case IA_MODEL_NLA:
    {
      if (include_HOD_GX == 1)
      {
        spdlog::debug("\x1b[90m{}\x1b[0m: HOD not implemented!", "kernel_for_C_gs_tomo_limber");
        exit(1);
      }

      double WRSD = 0.0;
      if (include_RSD_GS == 1)
      {
        const double chi_0 = f_K(ell/k);
        const double chi_1 = f_K((ell+1.)/k);
        const double a_0 = a_chi(chi_0);
        const double a_1 = a_chi(chi_1);
        WRSD = W_RSD(ell, a_0, a_1, nl);
      }
      
      const double C1ZS = IA_A1_Z1(a, growfac_a, ns);

      ans(0) = (WK-WS*C1ZS)*(WGAL*b1+WMAG*ell_prefactor*bmag+WRSD); // Pk
      ans(1) = (WK-WS*C1ZS)*WGAL; // 1loop
      break;
    }
    default:
    {
      exit(1);
    }
  }
  return ans*(chidchi.dchida/(fK*fK))*ell_prefactor2;
}

PYBIND11_MODULE(cosmolike_roman_real_FASTPT_interface, m)
{
  m.doc() = "CosmoLike Interface for roman-Y1 3x2pt Module";

    // --------------------------------------------------------------------
  // HELPER FUNCTIONS
  // --------------------------------------------------------------------
  m.def("kernel_for_C_ss_tomo_limber",
    &kernel_for_C_ss_tomo_limber,
    "Calculate the angular PS C_ss coefficients of various Pk terms",
    py::arg("a").none(false),
    py::arg("params").none(false)
  );

  m.def("kernel_for_C_gs_tomo_limber",
    &kernel_for_C_gs_tomo_limber,
    "Calculate the angular PS C_gs coefficients of various Pk terms",
    py::arg("a").none(false),
    py::arg("params").none(false)
  );


  // --------------------------------------------------------------------
  // INIT FUNCTIONS
  // --------------------------------------------------------------------
  m.def("init_ntable_lmax",
      &cosmolike_interface::init_ntable_lmax,
      "Init accuracy and sampling Boost (may slow down Cosmolike a lot)",
      (py::arg("lmax") = 75000).none(false)
    );
  
  m.def("init_accuracy_boost",
      &cosmolike_interface::init_accuracy_boost,
      "Init accuracy and sampling Boost (may slow down Cosmolike a lot)",
      (py::arg("accuracy_boost") = 1.0).none(false),
      (py::arg("integration_accuracy") = 0).none(false)
    );

  m.def("init_baryons_contamination",
      py::overload_cast<std::string, std::string>(
         &cosmolike_interface::init_baryons_contamination),
      "Init data vector contamination (on the matter power spectrum) with baryons",
      py::arg("sim").none(false),
      py::arg("allsims").none(false)
    );

  m.def("init_bias", 
      &cosmolike_interface::init_bias, 
      "Set the bias modeling choices",
      py::arg("bias_model").none(false),
      py::return_value_policy::move
    );

  m.def("init_binning",
      &cosmolike_interface::init_binning_real_space,
      "Init Bining related variables",
      py::arg("ntheta_bins").none(false).noconvert(),
      py::arg("theta_min_arcmin").none(false),
      py::arg("theta_max_arcmin").none(false)
    );

  m.def("init_cosmo_runmode",
      &cosmolike_interface::init_cosmo_runmode,
      "Init Run Mode (should we force the matter power spectrum to be linear)",
      py::arg("is_linear").none(false)
    );

  m.def("init_data_real",
      [](std::string cov, std::string mask, std::string data) {
        using namespace cosmolike_interface;
        init_data_Mx2pt_N<0,3>(cov, mask, data, {0, 1, 2});
      },
      "Load covariance matrix, mask (vec of 0/1s) and data vector",
      py::arg("COV").none(false),
      py::arg("MASK").none(false),
      py::arg("DATA").none(false)
    );

  m.def("init_ggl_exclude",
      &cosmolike_interface::init_ggl_exclude,
      "Set lens-source tomo bin pairs excluded in galaxy-galaxy lensing",
      py::arg("ggl_exclude").none(false)
    );

  m.def("init_IA",
      &cosmolike_interface::init_IA,
      "Init IA related options",
      py::arg("ia_model").none(false).noconvert(),
      py::arg("ia_redshift_evolution").none(false).noconvert(),
      py::arg("ia_code").none(false).noconvert()
    );

  m.def("init_probes",
      &cosmolike_interface::init_probes,
      "Init Probes (cosmic shear or 2x2pt or 3x2pt...)",
      py::arg("possible_probes").none(false)
    );

  m.def("initial_setup",
      &cosmolike_interface::initial_setup,
      "Initialize Cosmolike Variables to their Default Values"
    );

  m.def("init_redshift_distributions_from_files",
      &cosmolike_interface::init_redshift_distributions_from_files,
      "Init lens and source n(z) from files",
      py::arg("lens_multihisto_file").none(false),
      py::arg("lens_ntomo").none(false).noconvert(),
      py::arg("source_multihisto_file").none(false),
      py::arg("source_ntomo").none(false).noconvert()
    );

  m.def("init_survey_parameters",
      &cosmolike_interface::init_survey,
      "Init Survey Parameters",
      py::arg("surveyname").none(false),
      py::arg("area").none(false),
      py::arg("sigma_e").none(false)
    );

  m.def("read_redshift_distributions",
    &cosmolike_interface::read_redshift_distributions_from_files,
    "Read n(z) lens and source from files (same way as old cosmolike)",
    py::arg("lens_multihisto_file").none(false),
    py::arg("lens_ntomo").none(false).noconvert(),
    py::arg("source_multihisto_file").none(false),
    py::arg("source_ntomo").none(false).noconvert(),
    py::return_value_policy::move
  );
  
  m.def("set_IA_PS",
      &cosmolike_interface::set_IA_PS,
      "Set FPTIA if FASTPT is called",
      py::arg("IA_PS").none(false),
      py::arg("IA_k_min").none(false),
      py::arg("IA_k_max").none(false),
      py::arg("IA_k_cutoff").none(false),
      py::arg("N").none(false)
    );

  m.def("set_bias_PS",
      &cosmolike_interface::set_bias_PS,
      "Set FPTbias if FASTPT is called",
      py::arg("bias_PS").none(false),
      py::arg("bias_k_min").none(false),
      py::arg("bias_k_max").none(false),
      py::arg("bias_k_cutoff").none(false),
      py::arg("sigma4").none(false),
      py::arg("N").none(false)
    );

  m.def("init_lens_sample_size",
      &cosmolike_interface::set_lens_sample_size,
      "Set the lens number of tomo bins",
      py::arg("Ntomo").none(false).noconvert()
    );

  m.def("init_source_sample_size",
      &cosmolike_interface::set_source_sample_size,
      "Set the source number of tomo bins",
      py::arg("Ntomo").none(false).noconvert()
    );

  m.def("init_ntomo_powerspectra",
    &cosmolike_interface::init_ntomo_powerspectra,
    "Set the number of power spectra"
  );

  // --------------------------------------------------------------------
  // SET FUNCTIONS
  // --------------------------------------------------------------------
  m.def("set_distances",
      [](arma::Col<double> z, 
         arma::Col<double> chi)
      {
        spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_distances");
        using namespace cosmolike_interface;
        set_distances(z, chi);
        spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_distances");
      },
      "Set Distance (Cosmology)",
       py::arg("z").none(false),
       py::arg("chi").none(false),
       py::return_value_policy::move
    );

  m.def("set_cosmology",
      [](const double omega_matter,
         const double hubble,
         arma::Col<double> io_log10k_2D,
         arma::Col<double> io_z_2D, 
         arma::Col<double> io_lnP_linear,
         arma::Col<double> io_lnP_nonlinear,
         arma::Col<double> io_G,
         arma::Col<double> io_z_1D,
         arma::Col<double> io_chi)
      {
        spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_cosmology");
        using namespace cosmolike_interface;
        set_cosmological_parameters(omega_matter, hubble);
        set_linear_power_spectrum(io_log10k_2D,io_z_2D,io_lnP_linear);
        set_non_linear_power_spectrum(io_log10k_2D,io_z_2D,io_lnP_nonlinear);
        set_growth(io_z_2D,io_G);
        set_distances(io_z_1D,io_chi);
        spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_cosmology");
      },
      "Set Cosmological Paramters, Distance, Matter Power Spectrum, Growth Factor",
       py::arg("omegam").none(false),
       py::arg("H0").none(false),
       py::arg("log10k_2D").none(false),
       py::arg("z_2D").none(false),
       py::arg("lnP_linear").none(false),
       py::arg("lnP_nonlinear").none(false),
       py::arg("G").none(false),
       py::arg("z_1D").none(false),
       py::arg("chi").none(false),
       py::return_value_policy::move
    );

  m.def("set_baryon_pcs",
      [](arma::Mat<double> eigenvectors) {
        spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_baryon_pcs");
        cosmolike_interface::BaryonScenario::get_instance().set_pcs(eigenvectors);
        spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_baryon_pcs");
      },
      "Load baryonic principal components from numpy array",
       py::arg("eigenvectors").none(false)
    );

  m.def("set_nuisance_ia",
      &cosmolike_interface::set_nuisance_IA,
      "Set nuisance Intrinsic Aligment (IA) amplitudes",
      py::arg("A1").none(false),
      py::arg("A2").none(false),
      py::arg("B_TA").none(false),
      py::return_value_policy::move
    );

  m.def("set_nuisance_bias",
      &cosmolike_interface::set_nuisance_bias,
      "Set nuisance Bias Parameters",
      py::arg("B1").none(false),
      py::arg("B2").none(false),
      py::arg("B_MAG").none(false),
      py::arg("B3nl").none(false),
      py::arg("BK").none(false)
    );

  m.def("set_nuisance_shear_calib",
      &cosmolike_interface::set_nuisance_shear_calib,
      "Set nuisance shear calibration amplitudes",
      py::arg("M").none(false),
      py::return_value_policy::move
    );
  
  m.def("set_nuisance_shear_photoz",
      &cosmolike_interface::set_nuisance_shear_photoz,
      "Set nuisance shear photo-z bias amplitudes",
      py::arg("bias").none(false),
      py::return_value_policy::move
    );

  m.def("set_nuisance_clustering_photoz",
      &cosmolike_interface::set_nuisance_clustering_photoz,
      "Set nuisance clustering shear photo-z bias amplitudes",
      py::arg("bias"),
      py::return_value_policy::move
    );

  m.def("set_point_mass",
      [](arma::Col<double> PM) {
        cosmolike_interface::PointMass::get_instance().set_pm_vector(PM);
      },
      "Set the point mass amplitudes",
      py::arg("PMV").none(false)
    );

  m.def("set_log_level_debug", 
      []() {
        spdlog::set_level(spdlog::level::debug);
      },
      "Set the SPDLOG level to debug"
    );

  m.def("set_log_level_info", 
      []() {
        spdlog::set_level(spdlog::level::info);
      },
      "Set the SPDLOG level to info"
    );

  m.def("set_lens_sample",
      &cosmolike_interface::set_lens_sample,
      "Set the lens n(z) from a numpy n(z) histogram",
      py::arg("nofz").none(false)
    );

  m.def("set_source_sample",
      &cosmolike_interface::set_source_sample,
      "Set the source n(z) from a numpy n(z) histogram",
      py::arg("nofz").none(false)
    );

  // --------------------------------------------------------------------
  // reset FUNCTIONS
  // --------------------------------------------------------------------
  
  m.def("reset_bary_struct",
      &reset_bary_struct,
       "Set the Baryon Functions to not contaminate the MPS w/ Baryon effects"
    );

  // --------------------------------------------------------------------
  // COMPUTE FUNCTIONS (relevant for emulators)
  // --------------------------------------------------------------------
  m.def("compute_add_fpm_3x2pt_real_any_order",
      [](arma::Col<double> dv, 
         const int force_exclude_pm)->std::vector<double> 
      {
        using namespace cosmolike_interface;
        arma::Col<double> res;
        if (force_exclude_pm == 1) {
          res = compute_add_calib_and_set_mask_Mx2pt_N<0,3,0>(dv,{0, 1, 2});
        } 
        else {
          res = compute_add_calib_and_set_mask_Mx2pt_N<0,3,1>(dv,{0, 1, 2});
        }
        return arma::conv_to<std::vector<double>>::from(res);
      },
      "Add fast shear calibration parameters to the theoretical data vector.",
      py::arg("datavector").none(false),
      py::arg("force_exclude_pm").none(false),
      py::return_value_policy::move
    );

  m.def("compute_add_fpm_3x2pt_real_any_order_with_pcs",
      [](arma::Col<double> dv, 
         arma::Col<double> Q, 
         const int force_exclude_pm)->std::vector<double> 
      {
        using namespace cosmolike_interface;
        using stlvec = std::vector<double>;
        arma::Col<double> res;
        if (force_exclude_pm == 1) {
          res = compute_add_calib_and_set_mask_Mx2pt_N<0,3,0>(dv,{0, 1, 2});
        } 
        else {
          res = compute_add_calib_and_set_mask_Mx2pt_N<0,3,1>(dv,{0, 1, 2});
        }
        return arma::conv_to<stlvec>::from(compute_add_baryons_pcs(Q,res));
      },
      "Add fast shear calibration parameters to the theoretical data vector.",
      py::arg("datavector").none(false),
      py::arg("Q").none(false),
      py::arg("force_exclude_pm").none(false),
      py::return_value_policy::move
    );

  m.def("compute_data_vector_3x2pt_real_sizes",
      []()->std::vector<int> {
        using namespace cosmolike_interface;
        using namespace arma;
        using stlvec = std::vector<int>;
        return conv_to<stlvec>::from(compute_data_vector_Mx2pt_N_sizes<0,3>());
      },
      "Returns the data vector sizes of each 2pt correlation function",
      py::return_value_policy::move
    );

  // --------------------------------------------------------------------
  // COMPUTE FUNCTIONS
  // --------------------------------------------------------------------
  m.def("compute_data_vector_masked",
      []()->std::vector<double> {
        using namespace cosmolike_interface;
        using stlvec = std::vector<double>;
        return arma::conv_to<stlvec>::from(compute_Mx2pt_N_masked<0,3>({0,1,2}));
      },
      "Compute theoretical data vector. Masked dimensions are filled w/ zeros",
      py::return_value_policy::move
    );

  m.def("compute_data_vector_masked_with_baryon_pcs",
      [](std::vector<double> Q)->std::vector<double> {
        using namespace cosmolike_interface;
        using stlvec = std::vector<double>;
        arma::Col<double> res = compute_Mx2pt_N_masked<0,3>({0,1,2});
        return arma::conv_to<stlvec>::from(compute_add_baryons_pcs(Q,res));
      },
      "Compute theoretical data vector, including contributions from baryonic"
      " principal components. Masked dimensions are filled w/ zeros",
      py::arg("Q").none(false),
      py::return_value_policy::move
    );

  m.def("compute_chi2",
      [](arma::Col<double> datavector) {
        return cosmolike_interface::IP::get_instance().get_chi2(datavector);
      },
      "Compute $\\chi^2$ given a theory data vector input",
      py::arg("datavector").none(false),
      py::return_value_policy::move
    );

  m.def("compute_baryon_pcas",
      [](std::string scenarios, std::string allsims) {
        using namespace cosmolike_interface;
        BaryonScenario::get_instance().set_scenarios(allsims, scenarios);
        return compute_baryon_pcas_Mx2pt_N<0,3>({0,1,2});
      },
      "Compute baryonic principal components given a list of scenarios" 
      "that contaminate the matter power spectrum",
      py::arg("scenarios").none(false),
      py::arg("allsims").none(false),
      py::return_value_policy::move
    );

  // --------------------------------------------------------------------
  // Theoretical Cosmolike Functions
  // --------------------------------------------------------------------
  m.def("get_binning_real_space",
      // Why return an STL vector?
      // The conversion between STL vector and python np array is cleaner
      // arma:Col is cast to 2D np array with 1 column (not as nice!)
      []()->std::vector<double> {
        arma::Col<double> res = cosmolike_interface::get_binning_real_space();
        return arma::conv_to<std::vector<double>>::from(res);
      },
      "Get real space binning (theta bins)"
    );

  m.def("get_gs_redshift_bins",
      &cosmolike_interface::gs_bins,
      "Get galaxy-galaxy lensing redshift binning"
    );

  m.def("xi_pm_tomo",
      &cosmolike_interface::xi_pm_tomo_cpp,
      "Compute cosmic shear (real space) data vector at all tomographic"
      " and theta bins"
    );

  m.def("w_gammat_tomo",
      &cosmolike_interface::w_gammat_tomo_cpp,
      "Compute galaxy-galaxy lensing (real space) data vector at all"
      " tomographic and theta bins",
      py::return_value_policy::move
    );

  m.def("w_gg_tomo",
      &cosmolike_interface::w_gg_tomo_cpp,
      "Compute galaxy-galaxy clustering (real space) data vector at all"
      " tomographic and theta bins",
      py::return_value_policy::move
    );

  m.def("C_ss_tomo_limber",
      py::overload_cast<const double, const int, const int>(
        &cosmolike_interface::C_ss_tomo_limber_cpp
      ),
      "Compute shear-shear (fourier - limber) data vector at a single"
      " tomographic bin and ell value",
      py::arg("l").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::arg("nj").none(false).noconvert()
    );

  m.def("C_ss_tomo_limber",
      py::overload_cast<arma::Col<double>>(
        &cosmolike_interface::C_ss_tomo_limber_cpp
      ),
      "Compute shear-shear (fourier - limber) data vector at all tomographic"
      " bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("int_for_C_ss_tomo_limber",
      py::overload_cast<const double, const double, const int, const int>(
        &cosmolike_interface::int_for_C_ss_tomo_limber_cpp
      ),
      "Compute integrand for shear-shear (fourier - limber) data vector"
      " at a single tomographic bin and ell value",
      py::arg("a").none(false).noconvert(),
      py::arg("l").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::arg("nj").none(false).noconvert()
    );

  m.def("int_for_C_ss_tomo_limber",
      py::overload_cast<arma::Col<double>, arma::Col<double>>(
        &cosmolike_interface::int_for_C_ss_tomo_limber_cpp
      ),
      "Compute integrand shear-shear (fourier - limber) data vector at all" 
      " tomographic bins and many scale factor and ell (vectorized)",
      py::arg("a").none(false),
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("C_gs_tomo_limber",
      py::overload_cast<const double, const int, const int>(
        &cosmolike_interface::C_gs_tomo_limber_cpp
      ),
      "Compute shear-position (fourier - limber) data vector at a single"
      " tomographic bin and ell value",
      py::arg("l").none(false).noconvert(),
      py::arg("nl").none(false).noconvert(),
      py::arg("ns").none(false).noconvert()
    );

  m.def("C_gs_tomo_limber",
      py::overload_cast<arma::Col<double>>(
        &cosmolike_interface::C_gs_tomo_limber_cpp
      ),
      "Compute shear-position (fourier - limber) data vector at all tomographic"
      " bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("int_for_C_gs_tomo_limber",
      py::overload_cast<const double, const double, const int, const int>(
        &cosmolike_interface::int_for_C_gs_tomo_limber_cpp
      ),
      "Compute integrand for shear-position (fourier - limber) data vector"
      " at a single tomographic bin and ell value",
      py::arg("a").none(false).noconvert(),
      py::arg("l").none(false).noconvert(),
      py::arg("nl").none(false).noconvert(),
      py::arg("ns").none(false).noconvert()
    );

  m.def("int_for_C_gs_tomo_limber",
      py::overload_cast<arma::Col<double>, arma::Col<double>>(
        &cosmolike_interface::int_for_C_gs_tomo_limber_cpp
      ),
      "Compute integrand shear-shear (fourier - limber) data vector at all" 
      " tomographic bins and many scale factor and ell (vectorized)",
      py::arg("a").none(false),
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("C_gg_tomo_limber",
      py::overload_cast<arma::Col<double>>(
        &cosmolike_interface::C_gg_tomo_limber_cpp
      ),
      "Compute position-position (fourier - limber) data vector"
      " at all tomographic bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("C_gg_tomo",
      py::overload_cast<arma::Col<double>>(
        &cosmolike_interface::C_gg_tomo_cpp
      ),
      "Compute position-position (fourier - non-limber/limber) data vector"
      " at all tomographic bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  // --------------------------------------------------------------------
  // Miscellaneous
  // --------------------------------------------------------------------
  
  m.def("get_mask",
      // Why return an STL vector?
      // The conversion between STL vector and python np array is cleaner
      // arma:Col is cast to 2D np array with 1 column (not as nice!)
      []()->std::vector<int>{
        using namespace cosmolike_interface;
        arma::Col<int> res = IP::get_instance().get_mask();
        return arma::conv_to<std::vector<int>>::from(res);
      },
      "Get Mask Vector",
      py::return_value_policy::move
    );

  m.def("get_dv_masked",
      // Why return an STL vector?
      // The conversion between STL vector and python np array is cleaner
      // arma:Col is cast to 2D np array with 1 column (not as nice!)
      []()->std::vector<double> {
        using namespace cosmolike_interface;
        arma::Col<double> res = IP::get_instance().get_dv_masked();
        return arma::conv_to<std::vector<double>>::from(res);
      },
      "Get Mask Data Vector",
      py::return_value_policy::move
    );

  m.def("get_cov_masked",
      []()->arma::Mat<double> {
        return cosmolike_interface::IP::get_instance().get_cov_masked();
      },
      "Get Mask Covariance Matrix",
      py::return_value_policy::move
    );

  m.def("get_inv_cov_masked",
      []()->arma::Mat<double> {
        return cosmolike_interface::IP::get_instance().get_inv_cov_masked();
      },
      "Get Mask Covariance Matrix",
      py::return_value_policy::move
    );
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

int main()
{
  std::cout << "GOODBYE" << std::endl;
  exit(1);
}
