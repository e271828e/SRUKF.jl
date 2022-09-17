module Stages

using LinearAlgebra
using UnPack
using StaticArrays
using LazyArrays

using ..SRUT

export StatePropagator, StateCorrector

################################################################################
########################### StatePropagator ####################################

"""
    StatePropagator{LX, LW, S <: SquareRootUT}

Enables propagation of a vector Gaussian distribution \$x\$ of length `LX`
(state) through an arbitrary function \$x_{k+1} = f(x_{k}, w_k)\$, where \$w\$
is a vector Gaussian white noise of length `LW` (process noise).
"""
struct StatePropagator{LX, LW, S <: SquareRootUT}
    srut::S

    function StatePropagator(LX::Integer, LW::Integer)
        srut = SquareRootUT(LX, LW, LX)
        new{LX, LW, typeof(srut)}(srut)
    end

end

"""
    StatePropagator(LX::Integer, LW::Integer)
    StatePropagator(; LX::Integer, LW::Integer)

Construct a `StatePropagator` instance for a state of length `LX` and process
noise of length `LW`.
"""
StatePropagator(; LX::Integer, LW::Integer) = StatePropagator(LX, LW)

"""
    propagate!(sp::StatePropagator,
               x̄::AbstractVector{<:Real},
               S_δx::LowerTriangular{<:Real},
               S_δw::LowerTriangular{<:Real},
               f!::Function)


Propagate state `x`, subject to noise `w`, through the process dynamics given by
function `f!`, using `StatePropagator` `sp`.

# Arguments
- `sp::StatePropagator`: `StatePropagator{LX, LW}`.
- `x̄::AbstractVector{<:Real}`: State vector mean, `length(x̄) == LX`.
- `S_δx::LowerTriangular{<:Real}`: Square root (lower triangular Cholesky
  factor) of the state vector covariance matrix, `size(S_δx) == (LX, LX)`.
- `S_δw::LowerTriangular{<:Real}`:: Square root (lower triangular Cholesky
  factor) of the process noise covariance matrix, `size(S_δw) == (LW, LW)`.
- `f!::Function`: Process dynamics function. Its signature is `f!(x1, x0, w)`,
  where `x0` and `x1` are respectively the initial and the transformed state
  vectors (`x1` is updated in place).

This function mutates `sp`, `x̄` and `S_δx`.

If `P_δx` and `P_δw` are respectively the state and noise covariance matrices,
`S_δx` and `S_δw` may be initialized as follows:
```
using LinearAlgebra

S_δx = cholesky(P_δx).L
S_δw = cholesky(P_δw).L
```
"""
function propagate!(sp::StatePropagator, x̄::AbstractVector{<:Real},
                    S_δx::LowerTriangular{<:Real},
                    S_δw::LowerTriangular{<:Real},
                    f!::Function)

    SRUT.transform!(sp.srut, x̄, S_δx, S_δw, f!)
    assign!(x̄, S_δx, sp.srut)

end

@noinline function assign!(z̄::AbstractVector{<:Real},
                           S_δz::LowerTriangular{<:Real},
                           srut::SquareRootUT)

    copy!(z̄, srut.z̄)
    copy!(S_δz, srut.S_δz)
end

################################################################################
######################## StateCorrector ##################################

"""
    StateCorrector{LY, LX, LV, S <: SquareRootUT}

Enables correction of a vector Gaussian distribution \$x\$ of length `LX`
(state) from a vector sample \$\\tilde{y}\$ of length `LY` (measurement),
related to \$x\$ by an arbitrary function \$\\tilde{y} = h(x,v)\$, where \$w\$
is a vector Gaussian white noise of length `LV` (measurement noise).
"""
struct StateCorrector{LY, LX, LV, S <: SquareRootUT}
    srut::S
    δỹ::SizedVectorF64{LY} #innovation vector
    δx::SizedVectorF64{LX} #correction vector
    K::SizedMatrixF64{LX, LY} #Kalman gain
    U::SizedMatrixF64{LX, LY} #cache for K*S_δy

    function StateCorrector(LY::Integer, LX::Integer, LV::Integer)
        srut = SquareRootUT(LX, LV, LY) #input / noise / transformed lengths
        δỹ = zeros(SizedVectorF64{LY})
        δx = zeros(SizedVectorF64{LX})
        K = zeros(SizedMatrixF64{LX, LY})
        U = zeros(SizedMatrixF64{LX, LY})
        new{LY, LX, LV, typeof(srut)}(srut, δỹ, δx, K, U)
    end

end

StateCorrector(; LY::Integer, LX::Integer, LW::Integer) = StateCorrector(LY, LX, LW)

function update!(sc::StateCorrector,
                x̄::AbstractVector{<:Real}, #state mean
                S_δx::LowerTriangular{<:Real}, #state SR-covariance
                S_δv::LowerTriangular{<:Real}, #measurement noise SR-covariance
                ỹ::AbstractVector{<:Real}, #measurement sample
                h!::Function)  #measurement equation

    @unpack srut, δỹ, δx, K, U = sc

    SRUT.transform!(srut, x̄, S_δx, S_δv, h!)
    apply!(x̄, S_δx, δx, δỹ, K, U, ỹ, srut)

end

@noinline function apply!(x̄::AbstractVector{<:Real},
                           S_δx::LowerTriangular{<:Real},
                           δx::AbstractVector{<:Real},
                           δỹ::AbstractVector{<:Real},
                           K::AbstractMatrix{<:Real},
                           U::AbstractMatrix{<:Real},
                           ỹ::AbstractVector{<:Real},
                           srut::SquareRootUT)

    ȳ = srut.z̄
    S_δy = srut.S_δz
    P_δxy = srut.P_δxz

    #compute the Kalman gain, given by: K = P_δxy / P_δy. the right division
    #operator expects a factorization as its second argument. we need to provide
    #a Cholesky factorization for P_δy. this is trivial to construct: since S_δy
    #is a LowerTriangular, C_δz = Cholesky(S_δz) does the trick, yielding a
    #Cholesky instance with uplo = 'L'
    copy!(K, P_δxy)
    C_δy = Cholesky(LowerTriangular(S_δy.data.data))
    rdiv!(K.data, C_δy) #K now holds its final value

    #compute innovation and correction
    for i in eachindex(δỹ)
        δỹ[i] = ỹ[i] - ȳ[i]
    end
    mul!(δx, K, δỹ)
    #normalized innovation: δỹ ./ diag(P_δy)

    #update state mean
    for i in eachindex(x̄)
        x̄[i] += δx[i]
    end

    #update state SR-covariance
    mul!(U, K, S_δy)
    C_δx = Cholesky(S_δx)
    for u in eachcol(U)
        lowrankdowndate!(C_δx, u) #mutates S_δx
    end

end

end #module