using LinearAlgebra
import Ket
import QuantumNPA
import Dualization
import JuMP
import Hypatia

"""
    tsirelson_bound_postselection(V::Matrix, S::Matrix, scenario, level; verbose::Bool = false, dualize::Bool = false, solver = Hypatia.Optimizer{_solver_type(T)})

Upper bounds the Tsirelson bound of a post-selection game `V`, `S`, written in Collins-Gisin notation.
`scenario` is a tuple detailing the number of inputs and outputs, in the order (oa, ob, ia, ib).
`level` is an integer or a string like "1 + A B" determining the level of the NPA hierarchy.
`verbose` determines whether solver output is printed.
`dualize` determines whether the dual problem is solved instead. WARNING: This is critical for performance, and the correct choice depends on the solver.
"""
function tsirelson_bound_postselection(
    V::Matrix{T},
    S::Matrix{T},
    scenario,
    level;
    verbose = false,
    dualize = false,
    solver = Hypatia.Optimizer{Ket._solver_type(T)}
) where {T<:Real}
    sT = Ket._solver_type(T)
    V = convert(AbstractMatrix{sT}, V)
    S = convert(AbstractMatrix{sT}, S)

    if level == 1
        return _tsirelson_bound_postselection_manual(V, S, scenario, false; verbose, dualize = !dualize, solver)
    elseif level == "1 + A B" || level == "1+ A B" || level == "1 +A B" || level == "1+A B"
        return _tsirelson_bound_postselection_manual(V, S, scenario, true; verbose, dualize = !dualize, solver)
    end

    oa, ob, ia, ib = scenario
    A = QuantumNPA.projector(1, 1:oa-1, 1:ia)
    B = QuantumNPA.projector(2, 1:ob-1, 1:ib)
    aind(a, x) = 1 + a + (x - 1) * (oa - 1)
    bind(b, y) = 1 + b + (y - 1) * (ob - 1)

    model = JuMP.GenericModel{sT}()
    if dualize
        JuMP.set_optimizer(model, Dualization.dual_optimizer(solver; coefficient_type = T))
    else
        JuMP.set_optimizer(model, solver)
    end
    !verbose && JuMP.set_silent(model)

    G_basis = QuantumNPA.npa_moment([vec(A); vec(B)], level)
    mons = QuantumNPA.monomials(G_basis)
    JuMP.@variable(model, var[mons])
    dG = size(G_basis)[1]
    G = Matrix{typeof(1 * first(var))}(undef, dG, dG)
    for i ∈ eachindex(G)
        G[i] = 0
    end
    for m ∈ mons
        Ket._jump_muladd!(G, G_basis[m], var[m])
    end
    JuMP.@constraint(model, G in JuMP.PSDCone())

    bell_functional =
        sum(V[aind(a, x), bind(b, y)] * var[A[a, x]*B[b, y]] for a = 1:oa-1, b = 1:ob-1, x = 1:ia, y = 1:ib)
    bell_functional += sum(V[aind(a, x), 1] * var[A[a, x]] for a = 1:oa-1, x = 1:ia)
    bell_functional += sum(V[1, bind(b, y)] * var[B[b, y]] for b = 1:ob-1, y = 1:ib)
    bell_functional += V[1, 1] * var[QuantumNPA.Id]

    post = sum(S[aind(a, x), bind(b, y)] * var[A[a, x]*B[b, y]] for a = 1:oa-1, b = 1:ob-1, x = 1:ia, y = 1:ib)
    post += sum(S[aind(a, x), 1] * var[A[a, x]] for a = 1:oa-1, x = 1:ia)
    post += sum(S[1, bind(b, y)] * var[B[b, y]] for b = 1:ob-1, y = 1:ib)
    post += S[1, 1] * var[QuantumNPA.Id]

    JuMP.@constraint(model, post == 1)
    JuMP.@objective(model, Max, bell_functional)

    JuMP.optimize!(model)

    G = JuMP.value.(G)
    dq1 = 1 + ia * (oa - 1) + ib * (ob - 1)
    Γ = G[1:dq1, 1:dq1] / JuMP.value(var[QuantumNPA.Id])
    offset_a = ia * (oa - 1)
    behaviour = [Γ[1, 1] Γ[1, offset_a+2:end]'; Γ[1, 2:offset_a+1] Γ[2:offset_a+1, offset_a+2:end]]

    JuMP.is_solved_and_feasible(model) || @warn JuMP.raw_status(model)
    return JuMP.objective_value(model)::sT, behaviour::Matrix{sT}
end

function _tsirelson_bound_postselection_manual(V::Matrix{T}, S::Matrix{T}, scenario, include_ab::Bool; verbose, dualize, solver) where {T<:AbstractFloat}
    oa, ob, ia, ib = scenario
    alice_ops = ia * (oa - 1)
    bob_ops = ib * (ob - 1)
    dq1 = 1 + alice_ops + bob_ops
    dq1ab = dq1 + alice_ops * bob_ops
    model = JuMP.GenericModel{T}()
    dΓ = include_ab ? dq1ab : dq1
    JuMP.@variable(model, Γ[1:dΓ, 1:dΓ] in JuMP.PSDCone())
    ## normalization constraints
    #JuMP.@constraint(model, Γ[1, 1] == 1)
    for i ∈ 2:dq1
        JuMP.@constraint(model, Γ[1, i] == Γ[i, i])
    end
    ## q1 orthogonality constraints
    aind(a, x) = 1 + a + (x - 1) * (oa - 1)
    for x ∈ 1:ia, a1 ∈ 1:oa-1, a2 ∈ a1+1:oa-1
        JuMP.@constraint(model, Γ[aind(a1, x), aind(a2, x)] == 0)
    end
    bind(b, y) = 1 + alice_ops + b + (y - 1) * (ob - 1)
    for y ∈ 1:ib, b1 ∈ 1:ob-1, b2 ∈ b1+1:ob-1
        JuMP.@constraint(model, Γ[bind(b1, y), bind(b2, y)] == 0)
    end

    alice_marginal = Γ[1, 2:alice_ops+1]
    bob_marginal = Γ[1, alice_ops+2:dq1]
    correlation = Γ[2:alice_ops+1, alice_ops+2:dq1]
    behaviour = [Γ[1, 1] bob_marginal'; alice_marginal correlation]

    if include_ab
        ## first line of q1ab
        JuMP.@constraint(model, Γ[1, dq1+1:dq1ab] .== vec(correlation'))

        ## more normalization
        for i ∈ dq1+1:dq1ab
            JuMP.@constraint(model, Γ[1, i] == Γ[i, i])
        end

        function abind(a, b, x, y)
            apos = a + (x - 1) * (oa - 1)
            bpos = b + (y - 1) * (ob - 1)
            return dq1 + bpos + (apos - 1) * bob_ops
        end

        ## q1 × q1ab orthogonality constraints
        for x ∈ 1:ia
            for a1 ∈ 1:oa-1, a2 ∈ a1+1:oa-1, y ∈ 1:ib, b ∈ 1:ob-1
                JuMP.@constraint(model, Γ[aind(a1, x), abind(a2, b, x, y)] == 0)
                JuMP.@constraint(model, Γ[aind(a2, x), abind(a1, b, x, y)] == 0)
            end
        end
        for y ∈ 1:ib
            for b1 ∈ 1:ob-1, b2 ∈ b1+1:ob-1, x ∈ 1:ia, a ∈ 1:oa-1
                JuMP.@constraint(model, Γ[bind(b1, y), abind(a, b2, x, y)] == 0)
                JuMP.@constraint(model, Γ[bind(b2, y), abind(a, b1, x, y)] == 0)
            end
        end

        ## q1 × q1ab self equality constraints
        for x ∈ 1:ia, a ∈ 1:oa-1, y ∈ 1:ib, b ∈ 1:ob-1
            JuMP.@constraint(model, Γ[aind(a, x), abind(a, b, x, y)] == Γ[1, abind(a, b, x, y)])
        end
        for y ∈ 1:ib, b ∈ 1:ob-1, x ∈ 1:ia, a ∈ 1:oa-1
            JuMP.@constraint(model, Γ[bind(b, y), abind(a, b, x, y)] == Γ[1, abind(a, b, x, y)])
        end

        ## q1 × q1ab cross equality constraints
        for x1 ∈ 1:ia, x2 ∈ x1+1:ia
            for a1 ∈ 1:oa-1, a2 ∈ 1:oa-1, y ∈ 1:ib, b ∈ 1:ob-1
                JuMP.@constraint(model, Γ[aind(a1, x1), abind(a2, b, x2, y)] == Γ[aind(a2, x2), abind(a1, b, x1, y)])
            end
        end
        for y1 ∈ 1:ib, y2 ∈ y1+1:ib
            for b1 ∈ 1:ob-1, b2 ∈ 1:ob-1, x ∈ 1:ia, a ∈ 1:oa-1
                JuMP.@constraint(model, Γ[bind(b1, y1), abind(a, b2, x, y2)] == Γ[bind(b2, y2), abind(a, b1, x, y1)])
            end
        end

        ## q1ab × q1ab cross equality constraints
        for x ∈ 1:ia, a ∈ 1:oa-1
            for y1 ∈ 1:ib, y2 ∈ y1+1:ib, b1 ∈ 1:ob-1, b2 ∈ 1:ob-1
                JuMP.@constraint(
                    model,
                    Γ[abind(a, b1, x, y1), abind(a, b2, x, y2)] == Γ[bind(b1, y1), abind(a, b2, x, y2)]
                )
            end
        end
        for y ∈ 1:ib, b ∈ 1:ob-1
            for x1 ∈ 1:ia, x2 ∈ x1+1:ia, a1 ∈ 1:oa-1, a2 ∈ 1:oa-1
                JuMP.@constraint(
                    model,
                    Γ[abind(a1, b, x1, y), abind(a2, b, x2, y)] == Γ[aind(a1, x1), abind(a2, b, x2, y)]
                )
            end
        end

        ## q1ab × q1ab orthogonality constraints
        for x ∈ 1:ia, a1 ∈ 1:oa-1, a2 ∈ a1+1:oa-1
            for y1 ∈ 1:ib, y2 ∈ 1:ib, b1 ∈ 1:ob-1, b2 ∈ 1:ob-1
                JuMP.@constraint(model, Γ[abind(a1, b1, x, y1), abind(a2, b2, x, y2)] == 0)
            end
        end
        for y ∈ 1:ib, b1 ∈ 1:ob-1, b2 ∈ b1+1:ob-1
            for x1 ∈ 1:ia, x2 ∈ 1:ia, a1 ∈ 1:oa-1, a2 ∈ 1:oa-1
                JuMP.@constraint(model, Γ[abind(a1, b1, x1, y), abind(a2, b2, x2, y)] == 0)
            end
        end
    end

    bell_functional = dot(V, behaviour)
    post = dot(S, behaviour)

    JuMP.@constraint(model, post == 1)
    JuMP.@objective(model, Max, bell_functional)

    if dualize
        JuMP.set_optimizer(model, Dualization.dual_optimizer(solver; coefficient_type = T))
    else
        JuMP.set_optimizer(model, solver)
    end
    !verbose && JuMP.set_silent(model)

    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || @warn JuMP.raw_status(model)
    behaviour = JuMP.value.(behaviour)
    behaviour ./= behaviour[1]
    return JuMP.objective_value(model)::T, behaviour::Matrix{T}
end
