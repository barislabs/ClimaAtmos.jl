#=
Offline generator for the prognostic sea salt emission parameters: fits the
3-mode lognormal to the Gong (2003) spectrum, integrates the ClimaParams-stored
(rounded) modes into per-bin flux scales, and renders the docs figure
src/assets/gong_ln3_modes.png (gitignored; rebuilt by docs/make.jl). The
spectra and bin integrator live in the shared SeaSaltFitSkill module, which the
fit-skill unit test also exercises.
=#

import ClimaParams as CP
using CairoMakie

include(
    joinpath(
        @__DIR__,
        "..",
        "test",
        "parameterized_tendencies",
        "aerosols",
        "sea_salt_fit_skill.jl",
    ),
)
import .SeaSaltFitSkill as FS

#####
##### Parameters
#####

params = CP.get_parameter_values(
    CP.create_toml_dict(Float64),
    [
        "ssa_gong_logfit_mode1",
        "ssa_gong_logfit_mode2",
        "ssa_gong_logfit_mode3",
        "ssa_size_bin_divisions",
        "ssa_r_ref",
        "ssa_r80_per_dry",
        "seasalt_aerosol_density",
    ],
)
stored_modes = (
    Tuple(params.ssa_gong_logfit_mode1),
    Tuple(params.ssa_gong_logfit_mode2),
    Tuple(params.ssa_gong_logfit_mode3),
)
r_ref = params.ssa_r_ref
ρ_dry = params.seasalt_aerosol_density
r80_per_dry = params.ssa_r80_per_dry
r_hat_edges = params.ssa_size_bin_divisions ./ r_ref

#####
##### Fit: minimize the max relative error on a log-spaced radius grid
#####

# Dependency-free Nelder-Mead (reflection/expansion/contraction/shrink).
function nelder_mead(f, x0; iters = 20_000, init_step = 0.1)
    n = length(x0)
    simplex = [copy(x0)]
    for i in 1:n
        x = copy(x0)
        x[i] += init_step
        push!(simplex, x)
    end
    fvals = f.(simplex)
    for _ in 1:iters
        order = sortperm(fvals)
        simplex, fvals = simplex[order], fvals[order]
        centroid = sum(simplex[1:n]) / n
        xr = 2 * centroid - simplex[end]
        fr = f(xr)
        if fr < fvals[1]
            xe = 3 * centroid - 2 * simplex[end]
            fe = f(xe)
            simplex[end], fvals[end] = fe < fr ? (xe, fe) : (xr, fr)
        elseif fr < fvals[n]
            simplex[end], fvals[end] = xr, fr
        else
            xc = (centroid + simplex[end]) / 2
            fc = f(xc)
            if fc < fvals[end]
                simplex[end], fvals[end] = xc, fc
            else
                for i in 2:(n + 1)
                    simplex[i] = (simplex[i] + simplex[1]) / 2
                    fvals[i] = f(simplex[i])
                end
            end
        end
    end
    i_best = argmin(fvals)
    return simplex[i_best], fvals[i_best]
end

# Unconstrained fit variables: (log F, log r, log(σg - 1)) per mode.
to_modes(x) = ntuple(i -> (exp(x[3i - 2]), exp(x[3i - 1]), 1 + exp(x[3i])), 3)
from_modes(modes) = collect(
    Iterators.flatten((log(F), log(r), log(σ - 1)) for (F, r, σ) in modes),
)

r_grid = exp10.(range(log10(0.015), log10(10), 200))
target = FS.gong_spectrum.(r_grid, r80_per_dry)
max_rel_err(modes) = maximum(
    abs(FS.lognormal_spectrum(r, modes) / t - 1) for
    (r, t) in zip(r_grid, target)
)

guess = ((0.2, 0.05, 15.0), (60.0, 0.1, 1.8), (6.0, 0.8, 1.8))
x_fit, err_fit = nelder_mead(x -> max_rel_err(to_modes(x)), from_modes(guess))
fit_modes = to_modes(x_fit)
println("Fitted modes (F [m⁻² s⁻¹], r [r_ref], σg): ", fit_modes)
println("Max relative error vs Gong (2003): ", round(err_fit; sigdigits = 3))

#####
##### Per-bin flux scales from the ClimaParams (rounded) modes
#####

# moment = 0: number flux scale [m⁻² s⁻¹]; moment = 3 with the 4π/3 ρ_dry
# prefactor: dry-mass flux scale [kg m⁻² s⁻¹]. Both at u_10 = u_ref.
bin_moment(i, moment) = FS.bin_log_simpson(
    r_hat -> FS.lognormal_spectrum(r_hat, stored_modes) * (r_hat * r_ref)^moment,
    r_hat_edges[i],
    r_hat_edges[i + 1],
)
n_bins = length(r_hat_edges) - 1
number_scales = [bin_moment(i, 0) for i in 1:n_bins]
mass_scales = [4π / 3 * ρ_dry * bin_moment(i, 3) for i in 1:n_bins]
println("ssa_gong_logfit_bin_0M_flux: ", round.(number_scales; sigdigits = 4))
println("ssa_gong_logfit_bin_3M_flux: ", round.(mass_scales; sigdigits = 4))

#####
##### Figure: modes, their sum, and the Gong spectrum, as dF/dln r̂
#####

u_10 = 10.0  # m/s; the wind power law u_10^3.41 only scales the amplitude
wind_factor = u_10^3.41
r_plot = exp10.(range(log10(0.015), log10(10), 400))

edges_μm = r_hat_edges .* (r_ref * 1e6)  # r̂ is numerically μm for r_ref = 1 μm
fig = Figure(size = (900, 560))
ax = Axis(
    fig[1, 1],
    xscale = log10,
    yscale = log10,
    xticks = (collect(edges_μm), string.(edges_μm)),
    xlabel = "dry radius (μm)",
    ylabel = "dF/dln r̂ (m⁻² s⁻¹)",
    title = "3-mode lognormal fit of the Gong (2003) spectrum (u₁₀ = $u_10 m/s)",
)
# MERRA-2 bin number emissions from the fit, drawn as the bin-mean dF/dln r̂
# (k_i⁽⁰⁾ / Δln r̂) so the bar heights are directly comparable to the curves.
y_floor = 1e-2  # bar base; matches the axis lower limit
bin_bars = map(1:n_bins) do i
    lo, hi = r_hat_edges[i], r_hat_edges[i + 1]
    height = wind_factor * number_scales[i] / log(hi / lo)
    Rect2(lo, y_floor, hi - lo, height - y_floor)
end
poly!(
    ax,
    bin_bars,
    color = (:steelblue, 0.3),
    strokecolor = :steelblue,
    strokewidth = 1,
    label = "bin emission",
)
for (i, bar) in enumerate(bin_bars)
    text!(
        ax,
        sqrt(bar.origin[1] * (bar.origin[1] + bar.widths[1])),
        2 * y_floor;
        text = "bin $i",
        align = (:center, :bottom),
        color = :steelblue,
        fontsize = 13,
    )
end
for (i, (F, r_mode, σg)) in enumerate(stored_modes)
    lines!(
        ax,
        r_plot,
        [
            wind_factor * F * exp(-log(r / r_mode)^2 / (2 * log(σg)^2)) for
            r in r_plot
        ],
        label = "mode $i (r = $r_mode)",
    )
end
lines!(
    ax,
    r_plot,
    [wind_factor * r * FS.lognormal_spectrum(r, stored_modes) for r in r_plot],
    color = :black,
    linewidth = 3,
    label = "sum of 3 modes",
)
lines!(
    ax,
    r_plot,
    [wind_factor * r * FS.gong_spectrum(r, r80_per_dry) for r in r_plot],
    color = :gray,
    linestyle = :dot,
    linewidth = 3,
    label = "Gong (2003), Θ = 30",
)
ylims!(ax, 1e-2, 1e6)
axislegend(ax, position = :rt)
save(joinpath(@__DIR__, "src", "assets", "gong_ln3_modes.png"), fig)
println("Saved src/assets/gong_ln3_modes.png")
