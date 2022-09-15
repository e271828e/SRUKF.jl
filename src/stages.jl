module Stages

using LinearAlgebra
using UnPack
using StaticArrays
using LazyArrays

using ..SRUT

export StatePropagator, MeasurementProcessor

################################################################################
########################### StatePropagator ####################################

struct StatePropagator{LX, LW, S <: SquareRootUT}
    srut::S

    function StatePropagator(LX::Integer, LW::Integer)
        srut = SquareRootUT(LX, LW, LX)
        new{LX, LW, typeof(srut)}(srut)
    end

end

StatePropagator(; LX::Integer, LW::Integer) = StatePropagator(LX, LW)

"""
#Use StatePropagator sp to propagate the state x

#Given the process noise SR-covariance S_δw and the dynamic equation f!(y, x,
#w), the StatePropagator updates the state mean x̄ and SR-covariance S_δx.
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
######################## MeasurementProcessor ##################################

struct MeasurementProcessor{LY, LX, LV, S <: SquareRootUT}
    srut::S
    δỹ::SizedVectorF64{LY} #innovation vector
    δx::SizedVectorF64{LX} #correction vector
    K::SizedMatrixF64{LX, LY} #Kalman gain
    U::SizedMatrixF64{LX, LY} #cache for K*S_δy

    function MeasurementProcessor(LY::Integer, LX::Integer, LV::Integer)
        srut = SquareRootUT(LX, LV, LY) #input / noise / transformed lengths
        δỹ = zeros(SizedVectorF64{LY})
        δx = zeros(SizedVectorF64{LX})
        K = zeros(SizedMatrixF64{LX, LY})
        U = zeros(SizedMatrixF64{LX, LY})
        new{LY, LX, LV, typeof(srut)}(srut, δỹ, δx, K, U)
    end

end

MeasurementProcessor(; LY::Integer, LX::Integer, LW::Integer) = MeasurementProcessor(LY, LX, LW)

"""
Use MeasurementProcessor mp to apply a measurement ỹ to state x.

Given the measurement sample ỹ, the measurement noise SR-covariance S_δv and
the measurement equation h!(y, x, v), the MeasurementProcessor updates the
prior state mean x̄ and SR-covariance S_δx to their posterior values
"""
function update!(mp::MeasurementProcessor,
                x̄::AbstractVector{<:Real}, #state mean
                S_δx::LowerTriangular{<:Real}, #state SR-covariance
                S_δv::LowerTriangular{<:Real}, #measurement noise SR-covariance
                ỹ::AbstractVector{<:Real}, #measurement sample
                h!::Function)  #measurement equation

    @unpack srut, δỹ, δx, K, U = mp

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