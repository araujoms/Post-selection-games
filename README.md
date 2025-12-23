# Post-selection-games
Algorithms for computing local bounds and Tsirelson bounds of post-selection games


## Installation

First you need to install [Julia](https://docs.julialang.org/en/v1/manual/getting-started/). Then clone this repository with
```
git clone https://github.com/araujoms/Post-selection-games
```
Navigate to the folder, start Julia, and enter the package manager by typing `]`. Then activate the project with the command `activate .`, and install the dependencies with the command `instantiate`.

## Example usage
```julia
julia> includet("hardy_game.jl")

julia> V, S = hardy_game(2,2)
([0 0; 0 0;;; 0 0; 0 0;;;; 0 0; 0 0;;; 0 1; 0 0], [0 0; 1 0;;; 0 1; 0 0;;;; 0 1; 0 0;;; 0 1; 0 0])

julia> includet("local_bound_postselection.jl")

julia> local_bound_postselection(V,S)
0.5

julia> includet("tsirelson_bound_postselection.jl")

julia> Vcg, Scg = Ket.tensor_collinsgisin.((V,S))
([0.0 0.0 0.0; 0.0 0.0 0.0; 1.0 0.0 -1.0], [0.0 1.0 0.0; 1.0 -1.0 -1.0; 2.0 -1.0 -1.0])

julia> tsirelson_bound_postselection(Vcg, Scg, (2,2,2,2), "1 + A B")
(0.9999999836947221, [1.0 0.18110572976160183 0.5913533263386471; 0.35475616726642234 0.18110572966451877 0.3547561671018782; 0.11305714207305947 0.1130571417968083 0.06423156486866591])
```
