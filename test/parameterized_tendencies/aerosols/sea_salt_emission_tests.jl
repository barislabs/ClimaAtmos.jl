#=
Unit tests for the coupler-facing sea salt emission pieces in
  src/parameterized_tendencies/aerosols/sea_salt.jl

The pure functions (`wind_at_height`, `sslt_bin_emission_flux`) are what
ClimaCoupler evaluates per surface; the setter + tendency integration test
feeds synthetic per-bin fluxes through `set_sslt_surface_fluxes!` exactly
as the coupler does and checks the applied boundary tendency.
=#

using Test
import LinearAlgebra: dot
import ClimaAtmos as CA
import ClimaCore.Fields as Fields
# Reuse ClimaAtmos's own imports so the test needs no extra direct deps.
const CP = CA.CP
const SF = CA.SF

include(joinpath(@__DIR__, "..", "..", "test_helpers.jl"))

const FT = Float64

# The ssa_* ClimaParams keys land in ClimaParams v1.1.6; mirror them into a
# temporary override toml so the tests also run against older releases. This
# is the UNION of the prognostic-aerosol keys used across the sea salt
# feature branches (entries unknown to this branch's name_map are simply
# unread), so the same test file runs unchanged at every level of the
# branch stack.
const ssa_toml = joinpath(mktempdir(), "ssa_params.toml")
write(
    ssa_toml,
    """
    [ssa_size_bin_divisions]
    value = [0.03e-6, 0.1e-6, 0.5e-6, 1.5e-6, 5.0e-6, 10.0e-6]
    type = "float"
    [ssa_r_ref]
    value = 1.0e-6
    type = "float"
    [ssa_r80_per_dry]
    value = 2.0
    type = "float"
    [ssa_u_ref]
    value = 1.0
    type = "float"
    [ssa_gong_wind_exponent]
    value = 3.41
    type = "float"
    [ssa_gong_logfit_bin_3M_flux]
    value = [1.589e-16, 3.575e-15, 4.557e-14, 1.238e-13, 1.32e-13]
    type = "float"
    [ssa_residence]
    value = 47520.0
    type = "float"
    [ssa_gong_logfit_mode1]
    value = [0.2157, 0.05545, 17.02]
    type = "float"
    [ssa_gong_logfit_mode2]
    value = [60.93, 0.0914, 1.813]
    type = "float"
    [ssa_gong_logfit_mode3]
    value = [5.949, 0.776, 1.759]
    type = "float"
    [ssa_rh_cap]
    value = 0.99
    type = "float"
    [ssa_settling_courant_max]
    value = 0.5
    type = "float"
    [ssa_collection_efficiency]
    value = [5.2e-4, 5.2e-4, 0.42, 0.42, 0.94]
    type = "float"
    [air_viscosity_sutherland_reference]
    value = 1.716e-5
    type = "float"
    [air_viscosity_sutherland_temperature]
    value = 273.15
    type = "float"
    [air_viscosity_sutherland_constant]
    value = 110.4
    type = "float"
    [cunningham_slip_coefficients]
    value = [1.257, 0.4, 1.1]
    type = "float"
    [zhang_collection_prefactor]
    value = 3.0
    type = "float"
    [zhang_impaction_exponent]
    value = 2.0
    type = "float"
    [zhang_impaction_alpha_water]
    value = 100.0
    type = "float"
    [zhang_brownian_gamma_water]
    value = 0.56
    type = "float"
    """,
)

@testset "MOST wind at height" begin
    sfp = SF.Parameters.SurfaceFluxesParameters(
        CP.create_toml_dict(FT),
        CA.UF.GryanikParams,
    )
    u10(u★; L = FT(-50)) = CA.wind_at_height(FT(10), u★, L, sfp)
    @test u10(FT(0.3)) > 0
    @test u10(FT(0.6)) > u10(FT(0.3))         # monotone in friction velocity
    @test u10(FT(0)) == 0                     # calm surface
    # near-neutral L is guarded, both signs
    @test isfinite(u10(FT(0.3); L = FT(1e-30)))
    @test isfinite(u10(FT(0.3); L = FT(-1e-30)))
    # stable vs unstable ordering at fixed u★: stable profile is faster at 10 m
    @test u10(FT(0.3); L = FT(50)) > u10(FT(0.3); L = FT(-50))
end

@testset "Gong bin emission flux" begin
    scale, u_ref, wind_exp = FT(1.32e-13), FT(1), FT(3.41)
    F(u) = CA.sslt_bin_emission_flux(u, scale, u_ref, wind_exp)
    @test F(FT(1)) ≈ scale                    # flux scale is at u₁₀ = u_ref
    @test F(FT(0)) == 0
    # wind power law: doubling the wind multiplies the flux by 2^3.41
    @test F(FT(10)) / F(FT(5)) ≈ 2^wind_exp
end

include(joinpath(@__DIR__, "sea_salt_fit_skill.jl"))
@testset "Gong 3-mode lognormal fit skill" begin
    # Guard the ClimaParams-stored fit against the exact Gong (2003)
    # spectrum: as of the 2026 fit, max pointwise error is ~14% and the
    # per-bin number/mass flux errors are within ~6% (see
    # docs/sea_salt_emission_fit.jl); tolerances leave refit headroom.
    toml_dict = CP.create_toml_dict(FT; override_file = ssa_toml)
    s = SeaSaltFitSkill.skill(SeaSaltFitSkill.stored_params(toml_dict)...)
    @test s.max_pointwise < 0.2
    @test maximum(abs, s.number) < 0.08
    @test maximum(abs, s.mass) < 0.08
end

@testset "Emission setter and boundary tendency" begin
    config = CA.AtmosConfig(
        Dict(
            "config" => "column",
            "initial_condition" => "DecayingProfile",
            "microphysics_model" => "0M",
            "prognostic_aerosols" => ["SSLT"],
            "surface_setup" => "PrescribedSurface",
            "toml" => [ssa_toml],
            "strict_params" => false,
            "output_default_diagnostics" => false,
        ),
        job_id = "seasalt_emission_setter",
    )
    (; Y, p) = generate_test_simulation(config)
    FTc = eltype(Y)
    sfc_space = axes(Fields.level(Y.f, Fields.half))

    # Fluxes are zero-initialized (standalone run: no emission source), so
    # the tendency starts exactly zero.
    Yₜ = zero(Y)
    CA.aerosol_emission_tendency!(Yₜ, Y, p, FTc(0), p.atmos.seasalt)
    for name in CA.bin_names(p.atmos.seasalt)
        @test all(iszero, parent(getproperty(Yₜ.c, Symbol(:ρ, name))))
    end

    # Feed synthetic per-bin upward fluxes through the coupler-facing setter.
    bin_flux_values = FTc.((1e-16, 2e-15, 3e-14, 4e-14, 5e-14))
    bin_fluxes = map(bin_flux_values) do value
        flux = zeros(sfc_space)
        flux .= value
        flux
    end
    @test_throws ErrorException CA.set_sslt_surface_fluxes!(
        Y,
        p,
        bin_fluxes[1:2],
    )
    CA.set_sslt_surface_fluxes!(Y, p, bin_fluxes)

    # The stored covariant fluxes reproduce the physical magnitudes.
    for (i, value) in enumerate(bin_flux_values)
        sfc_flux = p.tracers.sslt_sfc_fluxes[i]
        physical = zeros(sfc_space)
        surface_ct3_unit = p.core.surface_ct3_unit
        @. physical = dot(sfc_flux, surface_ct3_unit)
        @test all(x -> x ≈ value, parent(physical))
    end

    # Emission is a source confined to the lowest cell, proportional to the
    # bin flux.
    Yₜ = zero(Y)
    CA.aerosol_emission_tendency!(Yₜ, Y, p, FTc(0), p.atmos.seasalt)
    n_levels = CA.Spaces.nlevels(axes(Y.c))
    lowest = map(CA.bin_names(p.atmos.seasalt)) do name
        ᶜρχₜ = getproperty(Yₜ.c, Symbol(:ρ, name))
        @test all(>(0), parent(Fields.level(ᶜρχₜ, 1)))
        for lev in 2:n_levels
            @test all(iszero, parent(Fields.level(ᶜρχₜ, lev)))
        end
        maximum(Fields.level(ᶜρχₜ, 1))
    end
    ratios = lowest ./ bin_flux_values
    @test all(r -> r ≈ ratios[1], ratios)
end
