using LinearAlgebra
import Ket
import Ket.Moment
import Dualization
import JuMP
import Hypatia
import SparseArrays as SA

"""
    bound_tsirelson_postselection(V::Matrix, S::Matrix, scenario, level; verbose::Bool = false, dualize::Bool = false, solver = Hypatia.Optimizer{_solver_type(T)})

Upper bounds the Tsirelson bound of a post-selection game `V`, `S`, written in Collins-Gisin notation.
`scenario` is a tuple detailing the number of inputs and outputs, in the order (oa, ob, ia, ib).
`level` is an integer or a string like "1 + A B" determining the level of the NPA hierarchy.
`verbose` determines whether solver output is printed.
`dualize` determines whether the dual problem is solved instead. WARNING: This is critical for performance, and the correct choice depends on the solver.
"""
function bound_tsirelson_postselection(
    vcg::Matrix{T},
    scg::Matrix{T},
    scenario::Tuple,
    level::Union{Integer,String};
    verbose::Bool = false,
    dualize::Bool = false,
    solver = Hypatia.Optimizer{Ket._solver_type(T)},
    solver_attributes = Pair[]
) where {T<:Real}
    sT = Ket._solver_type(T)
    vcg = convert(AbstractMatrix{sT}, vcg)
    scg = convert(AbstractMatrix{sT}, scg)

    level_int, additional = Moment.parse_level(Val(2), level)
    if level_int == 1 && (isempty(additional) || additional == [[1, 1]])
        include_ab = !isempty(additional)
        return _bound_tsirelson_postselection_manual(vcg, scg, scenario, include_ab; verbose, dualize = !dualize, solver)
    end

    outs = scenario[1:2]
    ins = scenario[3:4]
    max_length = 2 * max(level_int, maximum(length.(additional); init = 0))
    Q, behaviour =
        _npa_postselection(vcg, scg, Moment.Projector, Val(max_length), outs, ins, level_int, additional; verbose, dualize, solver, solver_attributes)
    return Q, behaviour
end

function _npa_postselection(
    vcg::Array{T,N},
    scg::Array{T,N},
    ::Type{O},
    ::Val{M},
    outs::NTuple{N,<:Integer},
    ins::NTuple{N,<:Integer},
    level_int::Int,
    additional::Vector{Vector{Int}};
    verbose,
    dualize,
    solver,
    solver_attributes
) where {T<:AbstractFloat,N,M,O<:Moment.Operator}
    model = JuMP.GenericModel{T}()
    MonomialType = Moment.Monomial{N,Moment.OperatorSequence{M,O}}
    S = Moment.generate_sequences(MonomialType, outs, ins, level_int, additional)
    monomial_dict, Γ_basis = Moment.moment_matrix(S)
    number_monomials = length(monomial_dict)
    JuMP.@variable(model, var[1:number_monomials])
    dΓ = size(Γ_basis[1], 1)
    Γ = Matrix{typeof(1 * first(var))}(undef, dΓ, dΓ)
    for i ∈ eachindex(Γ)
        Γ[i] = 0
    end
    Id = one(eltype(S))
    for i ∈ 1:number_monomials
        Ket._jump_muladd!(Γ, Γ_basis[i], var[i])
    end
    JuMP.@constraint(model, Symmetric(Γ) ∈ JuMP.PSDCone())

    behaviour_op = Moment.behaviour_operator(eltype(S), outs, ins)
    behaviour = Array{typeof(1 * first(var)),N}(undef, size(behaviour_op))
    for i ∈ 1:length(behaviour)
        behaviour[i] = var[monomial_dict[behaviour_op[i]]]
    end
    numerator = dot(vcg, behaviour)
    denominator = dot(scg, behaviour)

    JuMP.@constraint(model, denominator == 1)
    JuMP.@objective(model, Max, numerator)

    dualize && (solver = Dualization.dual_optimizer(solver; coefficient_type = _solver_type(T)))
    Ket._set_optimizer(model, solver, solver_attributes, verbose)
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || @warn JuMP.raw_status(model)
    return JuMP.objective_value(model)::T, (JuMP.value(behaviour)/JuMP.value(var[1]))::Array{T,N}
end

function _bound_tsirelson_postselection_manual(vcg::Matrix{T}, scg::Matrix{T}, scenario, include_ab::Bool; verbose, dualize, solver) where {T<:AbstractFloat}
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

    bell_functional = dot(vcg, behaviour)
    post = dot(scg, behaviour)

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
