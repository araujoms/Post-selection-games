using LinearAlgebra
import Ket
import Hypatia

"""
    bound_local_postselection(V::Array{T,4}, S::Array{T,4})

Computes the local bound of a post-selection game `V`, `S` written in probability notation.
"""
function bound_local_postselection(V::Array{T,4}, S::Array{T,4}) where {T<:Real}
    oa, ob, ia, ib = size(V)

    a_vec = Vector{Int}(undef, ia)
    b_vec = Vector{Int}(undef, ib)
    bound = zero(one(T) / one(T))
    for a = 0:oa^ia-1
        digits!(a_vec, a; base = oa)
        for b = 0:ob^ib-1
            digits!(b_vec, b; base = ob)
            p_win = T(0)
            p_post = T(0)
            for y = 1:ib, x = 1:ia
                p_win += V[a_vec[x]+1, b_vec[y]+1, x, y]
                p_post += S[a_vec[x]+1, b_vec[y]+1, x, y]
            end
            if p_post != 0
                if p_win == p_post
                    return p_win / p_post
                else
                    temp_bound = p_win / p_post
                    temp_bound > bound && (bound = temp_bound)
                end
            end
        end
    end
    return bound
end

"""
    bound_signalling_postselection(V::Array{T,4}, S::Array{T,4})

Computes the signalling bound of a post-selection game `V`, `S` written in probability notation.
"""
function bound_signalling_postselection(V::Array{T,4}, S::Array{T,4}) where {T<:Real}
    oa, ob, ia, ib = size(V)

    strategy = Vector{Int}(undef, ia * ib)
    bound = zero(one(T) / one(T))
    for index = 0:(oa*ob)^(ia*ib)-1
        digits!(strategy, index; base = oa * ob)
        p_win = T(0)
        p_post = T(0)
        for y = 1:ib, x = 1:ia
            a, b = divrem(strategy[(x - 1) * ib + y], ob)
            p_win += V[a + 1, b + 1, x, y]
            p_post += S[a + 1, b + 1, x, y]
        end
        if p_post != 0
            if p_win == p_post
                return p_win / p_post
            else
                temp_bound = p_win / p_post
                temp_bound > bound && (bound = temp_bound)
            end
        end
    end
    return bound
end

#polynomial complexity, but the overhead kills it for small scenarios
#the version above is faster despite exponential complexity
function bound_signalling_postselection_lp(
    V::Array{T,N2},
    S::Array{T,N2};
    verbose::Bool = false,
    solver = Hypatia.Optimizer{Ket._solver_type(T)},
    solver_attributes = Pair[]
) where {T<:Real,N2}
    N = div(N2, 2)
    scenario = size(V)
    outs = scenario[1:N]
    ins = scenario[N+1:2N]

    stT = Ket._solver_type(T)
    model = JuMP.GenericModel{stT}()
    JuMP.@variable(model, Pvec[1:prod(scenario)] ≥ 0)
    P = reshape(Pvec, scenario)
    JuMP.@constraint(model, dot(S, P) == 1)
    reference_normalization = sum(P[a, first(CartesianIndices(ins))] for a ∈ CartesianIndices(outs))
    for x ∈ CartesianIndices(ins)
        if x != first(CartesianIndices(ins))
            JuMP.@constraint(model, sum(P[a, x] for a ∈ CartesianIndices(outs)) == reference_normalization)
        end
    end
    JuMP.@objective(model, Max, dot(V, P))
    Ket._set_optimizer(model, solver, solver_attributes, verbose)
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || @warn JuMP.raw_status(model)
    return JuMP.objective_value(model)::stT
end
