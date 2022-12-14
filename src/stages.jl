module Stages

using LinearAlgebra
using UnPack
using StaticArrays
using LazyArrays

using ..SRUT

export Propagator, Updater
export UpdateLog, UpdateFlag, none, accepted, rejected

################################################################################
########################### Propagator ####################################

"""
    Propagator{LX, LW, S <: SquareRootUT}

Enables propagation of a vector Gaussian distribution \$x\$ of length `LX` (the
state) through an arbitrary function \$x_{k+1} = f(x_{k}, w_k)\$, where \$w\$ is
a vector Gaussian white noise of length `LW` (process noise).
"""
struct Propagator{LX, LW, S <: SquareRootUT}
    srut::S

    function Propagator(LX::Integer, LW::Integer)
        srut = SquareRootUT(LX, LW, LX)
        new{LX, LW, typeof(srut)}(srut)
    end

end

"""
    Propagator(LX::Integer, LW::Integer)
    Propagator(; LX::Integer, LW::Integer)

Construct a `Propagator` instance for a state of length `LX` and process
noise of length `LW`.
"""
Propagator(; LX::Integer, LW::Integer) = Propagator(LX, LW)

"""
    propagate!(sp::Propagator,
               x̄::AbstractVector{<:Real},
               S_δx::LowerTriangular{<:Real},
               S_δw::LowerTriangular{<:Real},
               f!::Function)


Propagate state `x`, subject to noise `w`, through the process dynamics given by
function `f!`, using `Propagator` `sp`.

# Arguments
- `sp::Propagator`: `Propagator{LX, LW}`.
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
function propagate!(sp::Propagator, x̄::AbstractVector{<:Real},
                    S_δx::LowerTriangular{<:Real},
                    S_δw::LowerTriangular{<:Real},
                    f!::Function)

    SRUT.transform!(sp.srut, x̄, S_δx, S_δw, f!)
    propagate!(x̄, S_δx, sp.srut)

end

@noinline function propagate!(  z̄::AbstractVector{<:Real},
                                S_δz::LowerTriangular{<:Real},
                                srut::SquareRootUT)

    copy!(z̄, srut.z̄)
    copy!(S_δz, srut.S_δz)
end

################################################################################
################################ Updater ##################################

@enum UpdateFlag begin
    none = 0
    accepted = 1
    rejected = 2
end

Base.@kwdef struct UpdateLog{LY, LX}
    flag::UpdateFlag = none
    ỹ::SVector{LY,Float64} = zeros(SVector{LY, Float64}) #measurement sample
    δỹ::SVector{LY,Float64} = zeros(SVector{LY, Float64}) #innovation
    δη::SVector{LY,Float64} = zeros(SVector{LY, Float64}) #normalized innovation
    δx::SVector{LX,Float64} = zeros(SVector{LX, Float64}) #state correction
    δξ::SVector{LX,Float64} = zeros(SVector{LX, Float64}) #normalized state correction
end

"""
    Updater{LY, LX, LV, S <: SquareRootUT}

Enables updating a vector Gaussian distribution \$x\$ of length `LX` (the state)
from a vector sample \$\\tilde{y}\$ of length `LY` (the measurement), related to
\$x\$ by an arbitrary function \$\\tilde{y} = h(x,v)\$, where \$w\$ is a vector
Gaussian white noise of length `LV` (measurement noise).
"""
struct Updater{LY, LX, LV, S <: SquareRootUT}
    srut::S
    δỹ::SizedVectorF64{LY} #innovation vector
    δη::SizedVectorF64{LY} #normalized innovation
    δx::SizedVectorF64{LX} #state correction
    δξ::SizedVectorF64{LX} #normalized state correction
    K::SizedMatrixF64{LX, LY} #Kalman gain
    M::SizedMatrixF64{LX, LY} #update matrix (K*S_δy)
    P_δy::SizedMatrixF64{LY, LY} #measurement covariance
    U_δy::SizedUpperTriangularF64{LY} #measurement SR-covariance transpose
    P_δx::SizedMatrixF64{LX, LX} #state covariance
    U_δx::SizedUpperTriangularF64{LX} #measurement SR-covariance transpose

    function Updater(LY::Integer, LX::Integer, LV::Integer)
        srut = SquareRootUT(LX, LV, LY) #input / noise / transformed lengths
        δỹ = zeros(SizedVectorF64{LY})
        δη = zeros(SizedVectorF64{LY})
        δx = zeros(SizedVectorF64{LX})
        δξ = zeros(SizedVectorF64{LX})
        K = zeros(SizedMatrixF64{LX, LY})
        M = zeros(SizedMatrixF64{LX, LY})
        P_δy = zeros(SizedMatrixF64{LY, LY})
        U_δy = SizedUpperTriangularF64{LY}(zeros(SizedMatrixF64{LY, LY}))
        P_δx = zeros(SizedMatrixF64{LX, LX})
        U_δx = SizedUpperTriangularF64{LX}(zeros(SizedMatrixF64{LX, LX}))
        new{LY, LX, LV, typeof(srut)}(srut, δỹ, δη, δx, δξ, K, M, P_δy, U_δy, P_δx, U_δx)
    end

end

Updater(; LY::Integer, LX::Integer, LV::Integer) = Updater(LY, LX, LV)

function update!(su::Updater{LY, LX},
                x̄::AbstractVector{<:Real}, #state mean
                S_δx::LowerTriangular{<:Real}, #state SR-covariance
                S_δv::LowerTriangular{<:Real}, #measurement noise SR-covariance
                ỹ::AbstractVector{<:Real}, #measurement sample
                h!::Function; #measurement equation
                σ_thr::Real = Inf) where {LY, LX} #normalized innovation acceptance/rejection threshold

    SRUT.transform!(su.srut, x̄, S_δx, S_δv, h!)
    flag = update!(x̄, S_δx, ỹ, su, σ_thr)

    return UpdateLog{LY, LX}(flag, ỹ, su.δỹ, su.δη, su.δx, su.δξ)

end

@noinline function update!( x̄::AbstractVector{<:Real}, #state mean
                            S_δx::LowerTriangular{<:Real}, #state SR-covariance
                            ỹ::AbstractVector{<:Real}, #measurement sample
                            su::Updater{LY, LX},
                            σ_thr::Real) where {LY, LX}

    @unpack srut, δỹ, δη, δx, δξ, K, M, P_δy, U_δy, P_δx, U_δx = su

    ########################## compute update ##################################

    ȳ = srut.z̄
    S_δy = srut.S_δz
    P_δxy = srut.P_δxz

    #measurement innovation
    for i in eachindex(δỹ)
        δỹ[i] = ỹ[i] - ȳ[i]
    end

    #measurement covariance
    transpose!(U_δy, S_δy)
    mul!(P_δy, S_δy, U_δy)

    #normalized innovation (for update check)
    for i in eachindex(δη)
        δη[i] = δỹ[i] / sqrt(P_δy[i,i])
    end

    #Kalman gain, given by: K = P_δxy / P_δy
    copy!(K, P_δxy)
    C_δy = Cholesky(LowerTriangular(S_δy.data.data))
    rdiv!(K.data, C_δy) #K now holds its final value

    #state correction
    mul!(δx, K, δỹ)

    #state covariance (non-essential, for update log only)
    S_δx_LX = LowerTriangular(SizedMatrix{LX, LX, Float64}(S_δx.data))
    transpose!(U_δx, S_δx_LX)
    mul!(P_δx, S_δx_LX, U_δx)

    #normalized state correction (non-essential, for update log only)
    for i in eachindex(δξ)
        δξ[i] = δx[i] / sqrt(P_δx[i,i])
    end

    ############################# check update #################################

    #check normalized innovation against acceptance threshold
    valid = true
    for c in δη
        (abs(c) < σ_thr) || (valid = false)
    end

    ############################### apply update ###############################

    if valid
        #update state mean
        for i in eachindex(x̄)
            x̄[i] += δx[i]
        end

        #update state SR-covariance
        mul!(M, K, S_δy)
        C_δx = Cholesky(S_δx)
        for m in eachcol(M)
            lowrankdowndate!(C_δx, m) #mutates S_δx
        end
    end

    return valid ? accepted : rejected

end




end #module