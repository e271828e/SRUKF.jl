module SRUKF

using UnPack
using Reexport
@reexport using BenchmarkTools

using StaticArrays
using LazyArrays
using LinearAlgebra
using LinearAlgebra: BlasFloat, BlasInt
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra.LAPACK: chklapackerror

export QRFactorization, SquareRootUT, StatePropagator, MeasurementProcessor

const SizedVectorF64{N} = SizedVector{N, Float64, Vector{Float64}}
const SizedMatrixF64{M, N} = SizedMatrix{M, N, Float64, 2, Matrix{Float64}}
const SizedLowerTriangularF64{N} = LowerTriangular{Float64, SizedMatrixF64{N, N}}


################################################################################
################################# QRFactorization #######################################

struct QRFactorization{M, N, W}
    A::SizedMatrixF64{M, N}
    τ::SizedVectorF64{N}
    work::SizedVectorF64{W}
    valid::Base.RefValue{Bool} #false: current results invalid, computation pending

    function QRFactorization(M::Integer, N::Integer)
        @assert M ≥ N "We only handle M x N matrices with M ≥ N"
        A = zeros(SizedMatrix{M, N})
        τ = zeros(SizedVector{N}) #in general min(m, n)
        W = qr_lwork_query(A, τ)
        work  = zeros(SizedVector{W})
        new{M, N, W}(A, τ, work, Ref(false))
    end
end

QRFactorization(t::Tuple{Integer, Integer}) = QRFactorization(t...)

function QRFactorization(A::AbstractMatrix{Float64})
    data = QRFactorization(size(A))
    qr!(data, A)
end

Base.isready(data::QRFactorization) = getfield(data,:valid)[]

Base.propertynames(::QRFactorization) = (:τ, :R)

Base.getproperty(data::QRFactorization, name::Symbol) = getproperty(data, Val(name))

@inline @generated function Base.getproperty(data::QRFactorization{M,N,W},
                                                ::Val{S}) where {M,N,W,S}
    if S === :R
        return :(isready(data) || error("Factorization not yet computed");
                 A = getfield(data, :A);
                 UpperTriangular(@view A[1:N, :]))
    elseif S === :τ
        return :(isready(data) || error("Factorization not yet computed");
                 getfield(data, :τ))
    else
        return :(error("QRFactorization has no property $S"))
    end
end

#call degqrf with lwork = -1 to query it for the optimal lwork for the given A
#and τ. this value will be returned in the first component of the work array we
#provide
function qr_lwork_query(A::SizedMatrixF64{M,N}, τ::SizedVectorF64{N}) where {M,N}

    work  = Vector{Float64}(undef, 1)
    info  = Ref{BlasInt}()
    lwork = BlasInt(-1)
    ccall((@blasfunc(dgeqrf), Base.liblapack_name), Cvoid,
            (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
            Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}),
            BlasInt(M), BlasInt(N), A, max(1,stride(A,2)), τ, work, lwork, info)

    chklapackerror(info[])

    lwork = max(BlasInt(1), BlasInt(real(work[1])))
    return lwork

end

function _qr!(data::QRFactorization{M,N,W}) where {M,N,W}

    A, τ, work = map(s -> getfield(data, s), (:A, :τ, :work))
    info  = Ref{BlasInt}()
    lwork = BlasInt(W)
    ccall((@blasfunc(dgeqrf), Base.liblapack_name), Cvoid,
            (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
            Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}),
            BlasInt(M), BlasInt(N), A, max(1,stride(A,2)), τ, work, lwork, info)

    chklapackerror(info[])

    getfield(data, :valid)[] = true

    return data

end

function LinearAlgebra.qr!(data::QRFactorization, M)
    A = getfield(data, :A)
    A .= M
    _qr!(data)
end


################################################################################
############################# SquareRootUT #####################################

Base.@kwdef struct UTParams
    α::Float64 = 1e-3
    β::Float64 = 2
    κ::Float64 = 0
end

Base.@kwdef struct UTWeights
    w_0m::Float64
    w_0c::Float64
    w_i::Float64
    γ::Float64
end

function UTWeights(params::UTParams, L::Integer)
    @unpack α, β, κ = params

    α² = α^2
    γ² = α² * (L + κ)
    γ = √γ²
    λ = γ² - L

    w_0m = λ / γ²
    w_0c = w_0m + (1 - α² + β)
    w_i = 1/(2γ²)

    return UTWeights(; w_0m, w_0c, w_i, γ)
end

struct SquareRootUT{LX, LW, LZ, LA, LR, LS, QR <: QRFactorization}
    ā::SizedVectorF64{LA} #mean of augmented variable
    z̄::SizedVectorF64{LZ} #mean of transformed variable
    _𝓪::SizedVectorF64{LA} #cache for augmented sigma point
    _𝔃::SizedVectorF64{LZ} #cache for transformed sigma point
    𝓐::SizedMatrixF64{LA, LS} #augmented sigma points
    δ𝓐::SizedMatrixF64{LA, LS} #unbiased augmented sigma points
    𝓩::SizedMatrixF64{LZ, LS} #transformed sigma points
    δ𝓩::SizedMatrixF64{LZ, LS} #unbiased transformed sigma points
    P_δxz::SizedMatrixF64{LX, LZ} #input and transformed variables cross-covariance
    S_δa::SizedLowerTriangularF64{LA} #SR-covariance of augmented variable
    S_δz::SizedLowerTriangularF64{LZ} #SR-covariance of transformed variable
    qrd::QR
    weights::UTWeights
    valid::Base.RefValue{Bool} #0: not computed, #1: valid results

    function SquareRootUT(LX::Integer, LW::Integer, LZ::Integer, params = UTParams())
        @assert all((LX, LW, LZ) .> 0)

        LA = LX + LW #length of augmented input
        LS = 2LA + 1 #number of sigma points
        LR = LS -1 #number of sigma points excluding 𝔁0

        ā = zeros(SizedVectorF64{LA})
        z̄ = zeros(SizedVectorF64{LZ})
        _𝓪 = zeros(SizedVectorF64{LA})
        _𝔃 = zeros(SizedVectorF64{LZ})
        𝓐 = zeros(SizedMatrixF64{LA, LS})
        δ𝓐 = zeros(SizedMatrixF64{LA, LS})
        𝓩 = zeros(SizedMatrixF64{LZ, LS})
        δ𝓩 = zeros(SizedMatrixF64{LZ, LS})
        P_δxz = zeros(SizedMatrixF64{LX, LZ})
        S_δa = SizedLowerTriangularF64{LA}(zeros(SizedMatrixF64{LA, LA}))
        S_δz = SizedLowerTriangularF64{LZ}(zeros(SizedMatrixF64{LZ, LZ}))
        qrd = QRFactorization(LR, LZ)
        weights = UTWeights(params, LA)
        valid = Ref(false)

        new{LX, LW, LZ, LA, LR, LS, typeof(qrd)}(
            ā, z̄, _𝓪, _𝔃, 𝓐, δ𝓐, 𝓩, δ𝓩, P_δxz, S_δa, S_δz, qrd, weights, valid)

    end

end

function SquareRootUT(; LX::Integer, LW::Integer, LZ::Integer, params = UTParams())
    SquareRootUT(LX, LW, LZ, params)
end

Base.propertynames(::SquareRootUT) = (:z̄, :S_δz, :P_δxz, :P_δz, :P_δa)

Base.getproperty(srut::SquareRootUT, name::Symbol) = getproperty(srut, Val(name))

@inline @generated function Base.getproperty(srut::SquareRootUT, ::Val{S}) where {S}

    ex_main = Expr(:block)

    ex_valid = quote
        if !getfield(srut, :valid)[]
            error("Transformation not yet performed")
        end
    end

    if S ∈ (:z̄, :S_δz, :P_δxz, :P_δz)
        push!(ex_main.args, ex_valid)
    end

    if S === :P_δa
        push!(ex_main.args, :(S_δa = srut.S_δa; P_δa = S_δa * S_δa'; return P_δa))
    elseif S === :P_δz
        push!(ex_main.args, :(S_δz = srut.S_δz;  P_δz = S_δz * S_δz'; return P_δz))
    else
        push!(ex_main.args, :(return getfield(srut, $(QuoteNode(S)))))
    end

    return ex_main

end

function transform!(srut::SquareRootUT{LX, LW, LZ, LA, LR, LS},
                    x̄::AbstractVector{<:Real},
                    S_δx::LowerTriangular{<:Real},
                    S_δw::LowerTriangular{<:Real},
                    f!::Function) where {LX, LW, LZ, LA, LR, LS}

    @assert length(x̄) == LX "Wrong input mean length"
    @assert size(S_δx) == (LX, LX) "Wrong square-root input covariance size"
    @assert size(S_δw) == (LW, LW) "Wrong square-root noise covariance size"

    @unpack ā, _𝓪, _𝔃, 𝓐, δ𝓐, 𝓩, δ𝓩, S_δa, qrd, weights, valid = srut
    @unpack w_0m, w_0c, w_i, γ = weights

    #avoid not-computed errors
    z̄ = getfield(srut, :z̄)
    S_δz = getfield(srut, :S_δz)
    P_δxz = getfield(srut, :P_δxz) #avoid not-computed error

    #assign augmented input mean blocks
    ā[1:LX] .= x̄
    ā[LX+1:end] .= 0 #noise

    S_δa[1:LX, 1:LX] = S_δx
    S_δa[LX+1:end, LX+1:end] = S_δw

    #generate unbiased augmented sigma points
    δ𝓐[:,1] .= 0
    δ𝓐1 = @view δ𝓐[:,2:LA+1]
    δ𝓐2 = @view δ𝓐[:,LA+2:end]
    for (s, δ𝓪1, δ𝓪2) in zip(eachcol(S_δa), eachcol(δ𝓐1), eachcol(δ𝓐2))
       _𝓪 .= γ .* s
       δ𝓪1 .= _𝓪
       δ𝓪2 .= .-_𝓪
    end

    #generate augmented sigma points
    𝓐 .= ā .+ δ𝓐

    #transform sigma points
    for (𝓪, 𝔃) in zip(eachcol(𝓐), eachcol(𝓩))
        𝔁 = @view 𝓪[1:LX]
        𝔀 = @view 𝓪[LX+1:end]
        f!(𝔃, 𝔁, 𝔀)
    end

    #compute the transformed mean and the unbiased transformed sigma points
    weights_m = (i == 0 ? w_0m : w_i for i in 0:LR)
    z̄, δ𝓩 = mean!(z̄, δ𝓩, 𝓩; weights = weights_m) #mutates z̄ and δ𝓩

    #define convenient views for SR-covariance and cross-covariance computation
    δ𝓧 = @view δ𝓐[1:LX, :]
    δ𝔁0 = @view δ𝓧[:, 1]
    δ𝓧i = @view δ𝓧[:, 2:end]
    δ𝔃0 = @view δ𝓩[:, 1]
    δ𝓩i = @view δ𝓩[:, 2:end]

    #form the QR factorization target, whose R factor is the upper triangular
    #Cholesky factor of P_δz, that is, the transpose of S_δz.
    w_i_sqrt = √w_i
    qr_target = @~ w_i_sqrt .* δ𝓩i'

    #factorize and assign R' to S_δz
    qr!(qrd, qr_target)
    adjoint!(S_δz, qrd.R)

    #perform the Cholesky update/downdate for δ𝔃0
    w_0c_sqrt_abs = √abs(w_0c)
    _𝔃 .= w_0c_sqrt_abs .* δ𝔃0
    C_δz = Cholesky(S_δz) #LowerTriangular automatically yields uplo = 'L'
    w_0c >= 0 ? lowrankupdate!(C_δz, _𝔃) : lowrankdowndate!(C_δz, _𝔃) #mutates S_δz

    #compute cross-covariance
    @~ P_δxz .= w_0c .* δ𝔁0 .* δ𝔃0'
    for (δ𝔁, δ𝔃) in zip(eachcol(δ𝓧i), eachcol(δ𝓩i))
        @~ P_δxz .+= w_i .* δ𝔁 .* δ𝔃'
    end

    srut.valid[] = true

    return z̄, S_δz, P_δxz

end

function mean!(z̄::SizedVector{M}, Z::SizedMatrix{M,N};
                weights = Iterators.repeated(1/N, N)) where {M,N}
    z̄ .= 0
    for (z, w) in zip(eachcol(Z), weights)
        z̄ .+= w .* z
    end
    return z̄
end

function mean!(z̄::SizedVector{M}, δZ::SizedMatrix{M,N}, Z::SizedMatrix{M,N};
                weights = Iterators.repeated(1/N, N)) where {M,N}
    mean!(z̄, Z; weights)
    δZ .= Z .- z̄
    return z̄, δZ
end


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

    transform!(sp.srut, x̄, S_δx, S_δw, f!)
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

    transform!(srut, x̄, S_δx, S_δv, h!)
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