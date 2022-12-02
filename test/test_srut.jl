module TestSRUT

using Test
using LinearAlgebra
using Random
using UnPack
using StaticArrays

using SRUKF

export test_srut

function test_srut()
    @testset verbose = true "SRUT" begin
        @testset verbose = true "QR" begin test_qr() end
        @testset verbose = true "Transform" begin test_transform() end
    end
end

function test_qr()

    Random.seed!(0)
    A = randn(11,5)
    B = randn(11,5)
    A_copy = copy(A)
    B_copy = copy(B)

    #result from non-mutating LinearAlgebra method
    qr_A_ref = qr(A)
    qr_B_ref = qr(B)

    #instantiating from a target matrix A preallocates storage and directly
    #computes the factorization, A is preserved
    qr_test = QRFactorization(A)
    @test qr_test.R ≈ qr_A_ref.R
    @test A == A_copy

    #pass a new matrix B to an existing QRFactorization instance to factorize
    #it, B is preserved
    qr!(qr_test, B)
    @test qr_test.R ≈ qr_B_ref.R
    @test B == B_copy

    b = @benchmarkable qr!($qr_test, $A) setup = ($A .= $A_copy)
    @test run(b).allocs == 0

end


function test_transform()

    #basic correctness test on a linear transformation with equal input, noise
    #and output sizes, check for allocations when using normal Array inputs
    N = 10
    srut = SquareRootUT(N, N, N)

    function g!(z, x, w)
        @. z = 2*x + 1 + w
    end

    x̄ = ones(N)
    P_δx = diagm(N, N, ones(N))
    P_δw = 1.0 * Matrix(I, N, N)
    S_δx = cholesky(P_δx).L
    S_δw = cholesky(P_δw).L

    SRUT.transform!(srut, x̄, S_δx, S_δw, g!)

    #we know this linear transformation must yield the following:
    @unpack z̄, P_δz, P_δxz = srut
    @test z̄ ≈ 2x̄ .+ 1
    @test P_δz ≈ 4*P_δx .+ P_δw
    @test P_δxz ≈ 2I

    #check for allocations when using regular Array inputs
    @test @ballocated(SRUT.transform!($srut, $x̄, $S_δx, $S_δw, $g!)) == 0

    #more exhaustive correctness and performance tests on a non-linear function
    #with different sizes, check for allocations using SizedArray inputs
    NX = 5; NW = 2; NZ = 3
    srut = SquareRootUT(NX, NW, NZ)

    function f!(z, x, w)
        xr = @view x[1:NZ]
        @. z = 2*xr*(xr-1) + 1 + w[1] + w[end]
    end

    Random.seed!(0)
    A = randn(NX, NX)
    x̄ = randn(SizedVector{NX})
    P_δx = SizedMatrix{NX,NX}(A * A')
    P_δw = SizedMatrix{NW, NW}(1.0I)
    S_δx = cholesky(P_δx).L
    S_δw = cholesky(P_δw).L

    x̄_copy = copy(x̄)
    S_δx_copy = copy(S_δx)
    S_δw_copy = copy(S_δw)
    SRUT.transform!(srut, x̄, S_δx, S_δw, f!)

    #check that x, Sdx, Sdw are unmodified
    @test x̄ == x̄_copy
    @test S_δx == S_δx_copy
    @test S_δw == S_δw_copy

    @unpack ā, z̄, S_δa, S_δz, P_δa, P_δz, weights, 𝓩, δ𝓩 = srut
    @unpack w_0m, w_0c, w_i = weights

    #cross check the internally stored augmented state mean and covariance
    @test ā == vcat(x̄, zeros(NW))
    @test P_δa ≈ vcat(hcat(P_δx, zeros(NX,NW)), hcat(zeros(NW,NX), P_δw))

    #cross-check internally computed z̄ and P_δz against those obtained with
    #allocating computations from sigma-points
    𝔃0 = @view 𝓩[:, 1]
    𝓩i = @view 𝓩[:, 2:end]
    δ𝔃0 = @view δ𝓩[:, 1]
    δ𝓩i = @view δ𝓩[:, 2:end]
    @test z̄ ≈ w_0m * 𝔃0 .+ w_i * dropdims(sum(𝓩i, dims = 2), dims = 2)
    @test P_δz ≈ w_0c * δ𝔃0 * δ𝔃0' + w_i * δ𝓩i * δ𝓩i'

    #check for allocations when using SizedArray inputs
    b = @benchmarkable SRUT.transform!($srut, $x̄, $S_δx, $S_δw, $f!)
    results = run(b)
    @test results.allocs == 0
    # display(results)

end

#test a noiseless unscented transform
function test_noiseless()

    NX = 5; NW = 0; NZ = 3
    srut = SquareRootUT(NX, NW, NZ)

    function f!(z, x, w)
        xr = @view x[1:NZ]
        @. z = 2*xr*(xr-1) + 1
    end

    Random.seed!(0)
    A = randn(NX, NX)
    x̄ = randn(SizedVector{NX})
    P_δx = SizedMatrix{NX,NX}(A * A')
    P_δw = SizedMatrix{NW, NW}(1.0I)
    S_δx = cholesky(P_δx).L

    #here we need a zero-length LowerTriangular, we cannot use Cholesky
    S_δw = LowerTriangular(zeros(SMatrix{0,0,Float64}))

    x̄_copy = copy(x̄)
    S_δx_copy = copy(S_δx)
    S_δw_copy = copy(S_δw)
    SRUT.transform!(srut, x̄, S_δx, S_δw, f!)

    #check that x, Sdx, Sdw are unmodified
    @test x̄ == x̄_copy
    @test S_δx == S_δx_copy
    @test S_δw == S_δw_copy

    @unpack ā, z̄, S_δa, S_δz, P_δa, P_δz, weights, 𝓩, δ𝓩 = srut
    @unpack w_0m, w_0c, w_i = weights

    #cross check the internally stored augmented state mean and covariance
    @test ā == vcat(x̄, zeros(NW))
    @test P_δa ≈ vcat(hcat(P_δx, zeros(NX,NW)), hcat(zeros(NW,NX), P_δw))

    #cross-check internally computed z̄ and P_δz against those obtained with
    #allocating computations from sigma-points
    𝔃0 = @view 𝓩[:, 1]
    𝓩i = @view 𝓩[:, 2:end]
    δ𝔃0 = @view δ𝓩[:, 1]
    δ𝓩i = @view δ𝓩[:, 2:end]
    @test z̄ ≈ w_0m * 𝔃0 .+ w_i * dropdims(sum(𝓩i, dims = 2), dims = 2)
    @test P_δz ≈ w_0c * δ𝔃0 * δ𝔃0' + w_i * δ𝓩i * δ𝓩i'

    #check for allocations when using SizedArray inputs
    b = @benchmarkable SRUT.transform!($srut, $x̄, $S_δx, $S_δw, $f!)
    results = run(b)
    @test results.allocs == 0
    # display(results)

    return


end




function cholesky_qr_benchmark()

    #compare the cost of computing the upper factor U of S by:

    #a) constructing P by multiplication, then computing S from scratch by
    #Cholesky factorization

    #b) performing QR factorization of A, then extracting its R factor

    #it is almost always faster to redo the Cholesky factorization from scratch,
    #particularly for near-square matrices!

    Random.seed!(0)
    A = randn(2,10)
    P = A * A'

    chol_ref = cholesky!(Hermitian(P))
    U = chol_ref.U
    display(U)

    qr_test = QRFactorization(A')
    R = qr_test.R
    display(R)

    display(U'*U)
    display(R'*R)
    @test U'*U ≈ R'*R atol = 1e-12

    bchol = @benchmarkable ($P .= @~ $A * ($A)'; cholesky!(Hermitian($P)))
    bqr = @benchmarkable (qr!($qr_test)) setup = ($qr_test.A .= ($A)')

    display(run(bchol))
    display(run(bqr))

end

end