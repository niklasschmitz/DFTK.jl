"""
Power nonlinearity, with energy C ∫ρ^α where ρ is the density
"""
struct PowerNonlinearity
    C::Real
    α::Real
end
(P::PowerNonlinearity)(basis) = TermPowerNonlinearity(basis, P.C, P.α)

struct TermPowerNonlinearity <: Term
    basis::PlaneWaveBasis
    C::Real
    α::Real
end

function ene_ops(term::TermPowerNonlinearity, ψ, occ; ρ, kwargs...)
    basis = term.basis
    dVol = basis.model.unit_cell_volume / prod(basis.fft_size)

    E = term.C * sum(ρ .^ term.α) * dVol
    potential = @. term.C * term.α * ρ^(term.α-1)

    # In the case of collinear spin, the potential is spin-dependent
    ops = [RealSpaceMultiplication(basis, kpoint, potential[:, :, :, kpoint.spin])
           for kpoint in basis.kpoints]
    (E=E, ops=ops)
end


_pnl_kernel(C, α, ρ) = @. C * α * (α-1) * ρ^(α-2)

function compute_kernel(term::TermPowerNonlinearity, dρspin=nothing;
                        ρ::RealFourierArray, ρspin=nothing, kwargs...)
    @assert term.basis.model.spin_polarization in (:none, :spinless, :collinear)
    K = Diagonal(vec(_pnl_kernel(term.C, term.α, ρ.real)))

    # PNL kernel is independent of spin, so to apply it to (ρtot, ρspin)^T
    # and obtain the same contribution to Vα and Vβ the operator has the block structure
    #     ( K 0 )
    #     ( K 0 )
    n_spin = term.basis.model.n_spin_components
    n_spin == 1 ? K : [K 0I; K 0I]
end

function apply_kernel(term::TermPowerNonlinearity, dρ;
                      ρ, kwargs...)
    @assert term.basis.model.spin_polarization in (:none, :spinless, :collinear)
    kernel = from_real(term.basis, _pnl_kernel(term.C, term.α, ρ.real) .* dρ.real)
end
