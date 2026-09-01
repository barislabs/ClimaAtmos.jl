import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import ..Parameters as CAP


@generated sslt_state_names(::PrognosticSeaSalt{names}) where {names} =
    :($(map(n -> MatrixFields.FieldName(Symbol(:ρ, n)), names)))

"""
    wind_at_height(z, ustar, obukhov_length, sfp)

MOST point wind speed [m/s] at height `z` over water, recovered from the
friction velocity and Obukhov length via `SurfaceFluxes.compute_profile_value`
with COARE3 roughness, forced non-negative. Pure and pointwise, so it can be
broadcast over surface fields; coupler-facing (ClimaCoupler evaluates it with
the ocean surface's own `ustar`/`obukhov_length` to drive
[`sslt_bin_emission_flux`](@ref)).
"""
function wind_at_height(z::FT, ustar::FT, obukhov_length::FT, sfp) where {FT}
    roughness = SF.COARE3RoughnessParams{FT}()
    u = SF.compute_profile_value(
        sfp,
        safe_obukhov_length(obukhov_length),
        SF.momentum_roughness(roughness, ustar, sfp, nothing),
        z,
        ustar,
        zero(ustar),
        UF.MomentumTransport(),
        SF.PointValueScheme(),
    )
    return max(u, zero(u))
end

# Keep the MOST profile finite near neutral stratification.
safe_obukhov_length(L) =
    ifelse(L < zero(L), min(L, -eps(typeof(L))), max(L, eps(typeof(L))))

#####
##### Tendencies
#####

"""
    sslt_bin_emission_flux(u₁₀, mass_flux_scale, u_ref, wind_exp)

Upward sea salt mass emission flux [kg m⁻² s⁻¹] of one size bin over open
ocean via the [Gong2003](@cite) wind power law times the bin's
`mass_flux_scale`. ClimaCoupler evaluates it per surface with the ocean's
own [`wind_at_height`](@ref) and area-weights the result before handing it
to [`set_sslt_surface_fluxes!`](@ref).
"""
sslt_bin_emission_flux(u₁₀, mass_flux_scale, u_ref, wind_exp) =
    mass_flux_scale * (u₁₀ / u_ref)^wind_exp

"""
    set_sslt_surface_fluxes!(Y, p, bin_fluxes)

Coupler-facing setter: store the per-bin upward emission mass fluxes
`bin_fluxes` (a tuple of scalar surface Fields on the atmos bottom-face
space, one per bin ordered smallest to largest, physical magnitude
[kg m⁻² s⁻¹], positive up, already area-weighted over each cell's ocean
part) into `p.tracers.sslt_sfc_fluxes` as the covariant components of
surface-normal vectors, from which [`aerosol_emission_tendency!`](@ref)
builds the tracers' bottom boundary conditions. ClimaCoupler calls this
once per coupling step, after its per-surface flux computation, so
emission is held constant over a coupling step like the other surface
fluxes; the covariant/metric-factor convention stays on the ClimaAtmos
side. No-op without prognostic sea salt, so uncoupled runs keep the
zero-initialized fluxes (prognostic sea salt has no emission source
standalone).
"""
set_sslt_surface_fluxes!(Y, p, bin_fluxes) =
    set_sslt_surface_fluxes!(Y, p, bin_fluxes, p.atmos.seasalt)
set_sslt_surface_fluxes!(Y, p, bin_fluxes, ::Nothing) = nothing
function set_sslt_surface_fluxes!(Y, p, bin_fluxes, ::PrognosticSeaSalt)
    sfc_fluxes = values(p.tracers.sslt_sfc_fluxes)
    length(bin_fluxes) == length(sfc_fluxes) || error(
        "expected $(length(sfc_fluxes)) sea salt bin fluxes, \
         got $(length(bin_fluxes))",
    )
    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(Y.f), Fields.half)
    for (sfc_flux, flux) in zip(sfc_fluxes, bin_fluxes)
        @. sfc_flux = C3(flux * unit_basis_vector_data(C3, sfc_local_geometry))
    end
    return nothing
end

"""
    aerosol_emission_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)

Apply the per-bin emission fluxes cached in `p.tracers.sslt_sfc_fluxes`
(see [`set_sslt_surface_fluxes!`](@ref)) as bottom boundary conditions on
the grid-mean `Y.c.ρ<bin>` tracers, using [`boundary_tendency_scalar`](@ref),
and mirror the specific tendency onto each updraft tracer.
"""
function aerosol_emission_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)
    n_updrafts = n_mass_flux_subdomains(p.atmos.turbconv_model)
    fluxes = p.tracers.sslt_sfc_fluxes

    MatrixFields.unrolled_foreach(sslt_state_names(sslt)) do ρχ_name
        ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
        ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        btt = boundary_tendency_scalar(ᶜχ, fluxes[MatrixFields.extract_first(ρχ_name)])
        @. ᶜρχₜ -= btt

        for j in 1:n_updrafts
            ᶜχʲₜ = MatrixFields.get_field(
                Yₜ.c.sgsʲs.:($j),
                specific_tracer_name(ρχ_name),
            )
            @. ᶜχʲₜ -= specific(btt, p.precomputed.ᶜρʲs.:($$j))
        end
    end
    return nothing
end

"""
    aerosol_deposition_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)

Exponential decay of the grid-mean and updraft sea salt tracers with the
`ssa_residence` timescale (0.55 days, from AeroCom III). Uniform rate
over-deposits small bins and under-deposits large ones; separate wet and dry
deposition tendencies are forthcoming.
"""
function aerosol_deposition_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)
    (; turbconv_model) = p.atmos
    ap = CAP.prognostic_aerosol_params(p.params)

    λ = inv(ap.τ_ssa)
    n_updrafts = n_mass_flux_subdomains(turbconv_model)

    MatrixFields.unrolled_foreach(sslt_state_names(sslt)) do ρχ_name
        ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
        ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
        @. ᶜρχₜ -= λ * ᶜρχ

        for j in 1:n_updrafts
            χ_name = specific_tracer_name(ρχ_name)
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
            ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
            @. ᶜχʲₜ -= λ * ᶜχʲ
        end
    end
    return nothing
end
