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
