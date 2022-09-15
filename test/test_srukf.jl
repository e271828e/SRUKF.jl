module TestSRUKF

using Test
using LinearAlgebra
using Random
using UnPack
using StaticArrays

using SRUKF

export test_srukf

function test_srukf()
    @testset verbose = true "SRUKF" begin
        @testset verbose = true "QR Factorization" begin test_qr() end
        @testset verbose = true "SquareRootUT" begin test_srut() end
        @testset verbose = true "State Propagator" begin test_state_propagator() end
        @testset verbose = true "Measurement Processor" begin test_measurement_processor() end
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

    #accessing results of a pending factorization should warn
    qr_new = QRFactorization(11, 5) #just preallocates storage
    @test_throws ErrorException (qr_new.R)
    @test_throws ErrorException (qr_new.τ)

    b = @benchmarkable qr!($qr_test, $A) setup = ($A .= $A_copy)
    @test run(b).allocs == 0

end

function test_srut()

    #basic correctness test on a linear transformation with equal input, noise
    #and output sizes, check for allocations when using normal Array inputs
    N = 10
    srut = SquareRootUT(N, N, N)

    function g!(z, x, w)
        @. z = 2*x + 1 + w
    end

    #disallow accessing results when computation is pending to prevent mistakes
    @test_throws ErrorException srut.z̄
    @test_throws ErrorException srut.S_δz
    @test_throws ErrorException srut.P_δxz
    @test_throws ErrorException srut.P_δz

    x̄ = ones(N)
    P_δx = diagm(N, N, ones(N))
    P_δw = 1.0 * Matrix(I, N, N)
    S_δx = cholesky(P_δx).L
    S_δw = cholesky(P_δw).L

    SRUKF.transform!(srut, x̄, S_δx, S_δw, g!)

    #we know this linear transformation must yield the following:
    @unpack z̄, P_δz, P_δxz = srut
    @test z̄ ≈ 2x̄ .+ 1
    @test P_δz ≈ 4*P_δx .+ P_δw
    @test P_δxz ≈ 2I

    #check for allocations when using regular Array inputs
    @test @ballocated(SRUKF.transform!($srut, $x̄, $S_δx, $S_δw, $g!)) == 0

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
    SRUKF.transform!(srut, x̄, S_δx, S_δw, f!)

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
    b = @benchmarkable SRUKF.transform!($srut, $x̄, $S_δx, $S_δw, $f!)
    results = run(b)
    @test results.allocs == 0
    # display(results)

end

function test_state_propagator()

    N = 5
    sp = StatePropagator(N, N)

    function g!(z, x, w)
        @. z = 2*x + 1 + w
    end

    Random.seed!(1)
    A = rand(N, N)

    x̄_0 = ones(N)
    P_δx_0 = A* A'
    S_δx_0 = cholesky(P_δx_0).L
    P_δw = Matrix(I, N, N)
    S_δw = cholesky(P_δw).L

    x̄_1 = copy(x̄_0)
    S_δx_1 = copy(S_δx_0)

    SRUKF.propagate!(sp, x̄_1, S_δx_1, S_δw, g!)
    P_δx_1 = S_δx_1 * S_δx_1'

    @test x̄_1 ≈ 2x̄_0 .+ 1
    @test P_δx_1 ≈ 4*P_δx_0 .+ P_δw

    #check for allocations with built-in Array inputs
    b = @benchmarkable begin SRUKF.propagate!($sp, $x̄_1, $S_δx_1, $S_δw, $g!)
        setup = ($x̄_1 .= $x̄_0; $S_δx_1 .= $S_δx_0) end

    results = run(b)
    # display(results)
    @test results.allocs == 0

    #check for allocations with SizedArray inputs
    x̄_0 = SizedVector{N}(ones(N))
    P_δx_0 = SizedMatrix{N,N}(A* A')
    S_δx_0 = cholesky(P_δx_0).L
    P_δw = SizedMatrix{N,N}(I)
    S_δw = cholesky(P_δw).L

    x̄_1 = copy(x̄_0)
    S_δx_1 = copy(S_δx_0)

    b = @benchmarkable begin SRUKF.propagate!($sp, $x̄_1, $S_δx_1, $S_δw, $g!)
        setup = ($x̄_1 .= $x̄_0; $S_δx_1 .= $S_δx_0) end

    results = run(b)
    # display(results)
    @test results.allocs == 0

end

function test_measurement_processor()

    LY = 2
    LX = 3
    LV = 2
    mp = MeasurementProcessor(LY, LX, LV)

    function h!(y, x, w)
        y[1] = x[1] + w[1]
        y[2] = x[2] + w[2]
    end

    #synthetic measurement
    x̄_prior = ones(SizedVector{LX})
    P_δx_prior = SizedMatrix{LX,LX}(1.0I)
    S_δx_prior = cholesky(P_δx_prior).L
    P_δv = SizedMatrix{LV,LV}(diagm(LV, LV, [1, 4]))
    S_δv = cholesky(P_δv).L
    ỹ = SizedVector{2}([1.1, 1.1])

    x̄_post = copy(x̄_prior)
    S_δx_post = copy(S_δx_prior)

    SRUKF.update!(mp, x̄_post, S_δx_post, S_δv, ỹ, h!)
    P_δx_post = S_δx_post * S_δx_post'

    #for the states included in the measurement σ must decrease, for the other
    #one σ must remain unchanged
    @test P_δx_post[1,1] < P_δx_prior[1,1]
    @test P_δx_post[2,2] < P_δx_prior[2,2]
    @test P_δx_post[3,3] ≈ P_δx_prior[3,3]

    #measurement noise is smaller for the first state than for the second, so
    #its σ must decrease more and its mean should be pulled more closely towards
    #to the measurement sample (i.e., its residual should be smaller)
    @test P_δx_post[1,1] < P_δx_post[2,2]
    @test abs(ỹ[1] - x̄_post[1]) < abs(ỹ[2] - x̄_post[2])

    #check for allocations with SizedArray inputs
    b = @benchmarkable begin SRUKF.update!($mp, $x̄_post, $S_δx_post, $S_δv, $ỹ, $h!)
        setup = ($x̄_post .= $x̄_prior; $S_δx_post .= $S_δx_prior) end
    results = run(b)
    # display(results)
    @test results.allocs == 0

    x̄_prior = ones(LX)
    P_δx_prior = Matrix(I, LX,LX)
    S_δx_prior = cholesky(P_δx_prior).L
    P_δv = diagm(LV, LV, [1, 4])
    S_δv = cholesky(P_δv).L
    ỹ = [1.1, 1.1]

    x̄_post = copy(x̄_prior)
    S_δx_post = copy(S_δx_prior)

    #check for allocations with built-in Array inputs
    b = @benchmarkable begin SRUKF.update!($mp, $x̄_post, $S_δx_post, $S_δv, $ỹ, $h!)
        setup = ($x̄_post .= $x̄_prior; $S_δx_post .= $S_δx_prior) end
    results = run(b)
    display(results)
    @test results.allocs == 0

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