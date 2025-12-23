"""
    hardy_game(s::Int, k::Int)

Produces the (unnormalized) generalized Hardy game with `s` inputs and `k` outputs per party.
"""
function hardy_game(s::Int, k::Int)
    V = zeros(Int, k, k, s, s)
    S = zeros(Int, k, k, s, s)
    for b ∈ 2:k
        for a ∈ 1:b-1
            V[a, b, s, s] = 1
            S[a, b, s, s] = 1
            S[a, b, 1, s] = 1
            for x ∈ 1:s-1
                S[a, b, x+1, x] = 1
                S[b, a, x, x] = 1
            end
        end
    end

    return V, S
end
