module TestStages

using Test
using LinearAlgebra
using Random
using UnPack
using StaticArrays
using ComponentArrays

using SRUKF

export test_stages

function test_stages()
    @testset verbose = true "Filter Stages" begin
        @testset verbose = true "Propagator" begin test_propagator() end
        @testset verbose = true "Noiseless Propagator" begin test_propagator_noiseless() end
        @testset verbose = true "Updater" begin test_updater() end
    end
end

function test_allocs_sp(sp, x̄_0, S_δx_0, S_δw, g!)
    x̄_1 = copy(x̄_0)
    S_δx_1 = copy(S_δx_0)
    b = @benchmarkable begin Stages.propagate!($sp, $x̄_1, $S_δx_1, $S_δw, $g!)
        setup = ($x̄_1 .= $x̄_0; $S_δx_1 .= $S_δx_0) end
    results = run(b)
    # display(results)
    @test results.allocs == 0
end


function test_allocs_sc(sc, x̄_prior, S_δx_prior, S_δv, ỹ, h!)
    x̄_post = copy(x̄_prior)
    S_δx_post = copy(S_δx_prior)
    b = @benchmarkable begin Stages.update!($sc, $x̄_post, $S_δx_post, $S_δv, $ỹ, $h!)
        setup = ($x̄_post .= $x̄_prior; $S_δx_post .= $S_δx_prior) end
    results = run(b)
    display(results)
    @test results.allocs == 0
end


function test_propagator()

    N = 5
    sp = Propagator(N, N)

    function g!(z, x, w)
        @. z = 2*x + 1 + w
    end

    @testset verbose = true "Correctness" begin

        Random.seed!(1)
        A = rand(N, N)

        x̄_0 = ones(N)
        P_δx_0 = A* A'
        S_δx_0 = cholesky(P_δx_0).L
        P_δw = Matrix(I, N, N)
        S_δw = cholesky(P_δw).L

        x̄_1 = copy(x̄_0)
        S_δx_1 = copy(S_δx_0)
        Stages.propagate!(sp, x̄_1, S_δx_1, S_δw, g!)
        P_δx_1 = S_δx_1 * S_δx_1'

        #expected results for a Gaussian linear transformation
        @test x̄_1 ≈ 2x̄_0 .+ 1
        @test P_δx_1 ≈ 4*P_δx_0 .+ P_δw

    end

    @testset verbose = true "Allocations" begin

        #with built-in Array inputs
        Random.seed!(1)
        A = rand(N, N)

        x̄_0 = randn(N)
        P_δx_0 = A* A'
        S_δx_0 = cholesky(P_δx_0).L
        P_δw = Matrix(I, N, N)
        S_δw = cholesky(P_δw).L

        test_allocs_sp(sp, x̄_0, S_δx_0, S_δw, g!)

        #with SizedArray inputs
        x̄_0 = SizedVector{N}(randn(N))
        P_δx_0 = SizedMatrix{N,N}(A* A')
        S_δx_0 = cholesky(P_δx_0).L
        P_δw = SizedMatrix{N,N}(I)
        S_δw = cholesky(P_δw).L

        test_allocs_sp(sp, x̄_0, S_δx_0, S_δw, g!)

        #with ComponentArray inputs
        x̄_0 = ComponentVector(xa = 1.0, xb = 1.0, xc = 1.0, xd = 1.0, xe = 1.0)
        P_δx_axes = getaxes(x̄_0 * x̄_0')
        P_δx_0 = ComponentMatrix(A * A', P_δx_axes)
        S_δx_0 = cholesky(P_δx_0).L
        P_δw_axes = P_δx_axes
        P_δw = ComponentMatrix(Matrix(1.0I, N, N), P_δw_axes)
        S_δw = cholesky(P_δw).L

        test_allocs_sp(sp, x̄_0, S_δx_0, S_δw, g!)

    end

end

function test_propagator_noiseless()

    N = 5
    sp = Propagator(N, 0)

    function g!(z, x, w)
        @. z = 2*x + 1
    end

    @testset verbose = true "Correctness" begin

        Random.seed!(1)
        A = rand(N, N)

        x̄_0 = ones(N)
        P_δx_0 = A* A'
        S_δx_0 = cholesky(P_δx_0).L
        S_δw = LowerTriangular(zeros(SMatrix{0,0,Float64}))

        x̄_1 = copy(x̄_0)
        S_δx_1 = copy(S_δx_0)
        Stages.propagate!(sp, x̄_1, S_δx_1, S_δw, g!)
        P_δx_1 = S_δx_1 * S_δx_1'

        #expected results for a Gaussian linear transformation
        @test x̄_1 ≈ 2x̄_0 .+ 1
        @test P_δx_1 ≈ 4*P_δx_0

    end

    return

    @testset verbose = true "Allocations" begin

        #with built-in Array inputs
        Random.seed!(1)
        A = rand(N, N)

        x̄_0 = randn(N)
        P_δx_0 = A* A'
        S_δx_0 = cholesky(P_δx_0).L
        S_δw = LowerTriangular(zeros(SMatrix{0,0,Float64}))

        test_allocs_sp(sp, x̄_0, S_δx_0, S_δw, g!)

        #with SizedArray inputs
        x̄_0 = SizedVector{N}(randn(N))
        P_δx_0 = SizedMatrix{N,N}(A* A')
        S_δx_0 = cholesky(P_δx_0).L
        S_δw = LowerTriangular(zeros(SMatrix{0,0,Float64}))

        test_allocs_sp(sp, x̄_0, S_δx_0, S_δw, g!)

        #with ComponentArray inputs
        x̄_0 = ComponentVector(xa = 1.0, xb = 1.0, xc = 1.0, xd = 1.0, xe = 1.0)
        P_δx_axes = getaxes(x̄_0 * x̄_0')
        P_δx_0 = ComponentMatrix(A * A', P_δx_axes)
        S_δx_0 = cholesky(P_δx_0).L
        S_δw = LowerTriangular(zeros(SMatrix{0,0,Float64}))

        test_allocs_sp(sp, x̄_0, S_δx_0, S_δw, g!)

    end

end

function test_updater()

    LY = 2
    LX = 3
    LV = 2
    sc = Updater(LY, LX, LV)

    function h!(y, x, w)
        y[1] = x[1] + w[1]
        y[2] = x[2] + w[2]
    end

    @testset verbose = true "Correctness" begin

        x̄_prior = ones(SizedVector{LX})
        P_δx_prior = SizedMatrix{LX,LX}(1.0I)
        S_δx_prior = cholesky(P_δx_prior).L

        P_δv = SizedMatrix{LV,LV}(diagm(LV, LV, [1, 4]))
        S_δv = cholesky(P_δv).L

        x̄_post = copy(x̄_prior)
        S_δx_post = copy(S_δx_prior)

        σ_thr = 3

        #test measurement rejection
        ỹ = SizedVector{2}([10.0, 1.0])
        log = Stages.update!(sc, x̄_post, S_δx_post, S_δv, ỹ, h!; σ_thr)
        @test log.flag == rejected #measurement must have been rejected
        σ_δy = sqrt.(diag(sc.P_δy))
        @test sc.δỹ ≈ sc.δη .* σ_δy
        @test any(sc.δη .> 3)

        #test measurement acceptance
        ỹ = SizedVector{2}([1.1, 1.1])
        log = Stages.update!(sc, x̄_post, S_δx_post, S_δv, ỹ, h!; σ_thr)
        @test log.flag == accepted #measurement must have been processed

        P_δx_post = S_δx_post * S_δx_post'
        #for the states included in the measurement σ must decrease, for the other
        #one σ must remain unchanged
        @test P_δx_post[1,1] < P_δx_prior[1,1]
        @test P_δx_post[2,2] < P_δx_prior[2,2]
        @test P_δx_post[3,3] ≈ P_δx_prior[3,3]

        #measurement noise is smaller for the first state than for the second,
        #so its σ must decrease more and its measurement residual must be
        #smaller
        @test P_δx_post[1,1] < P_δx_post[2,2]
        @test abs(ỹ[1] - x̄_post[1]) < abs(ỹ[2] - x̄_post[2])

    end

    @testset verbose = true "Allocations" begin

        #check for allocations with SizedArray inputs
        x̄_prior = ones(SizedVector{LX})
        P_δx_prior = SizedMatrix{LX,LX}(1.0I)
        S_δx_prior = cholesky(P_δx_prior).L

        P_δv = SizedMatrix{LV,LV}(diagm(LV, LV, [1, 4]))
        S_δv = cholesky(P_δv).L
        ỹ = SizedVector{2}([1.1, 1.1])

        test_allocs_sc(sc, x̄_prior, S_δx_prior, S_δv, ỹ, h!)

        #check for allocations with built-in Array inputs
        x̄_prior = ones(LX)
        P_δx_prior = Matrix(I, LX,LX)
        S_δx_prior = cholesky(P_δx_prior).L

        P_δv = diagm(LV, LV, [1, 4])
        S_δv = cholesky(P_δv).L
        ỹ = [1.1, 1.1]

        test_allocs_sc(sc, x̄_prior, S_δx_prior, S_δv, ỹ, h!)

        #check for allocations with ComponentArray inputs
        x̄_prior = ComponentVector(xa = 1.0, xb = 1.0, xc = 1.0)
        P_δx_axes = getaxes(x̄_prior * x̄_prior')
        P_δx_prior = ComponentMatrix(Matrix(1.0I, LX,LX), P_δx_axes)
        S_δx_prior = cholesky(P_δx_prior).L

        ỹ = ComponentVector(ya = 1.1, yb = 1.1)
        P_δv_axes = getaxes(ỹ * ỹ')
        P_δv = ComponentMatrix(diagm(2, 2, [1, 4]), P_δv_axes)
        S_δv = cholesky(P_δv).L

        test_allocs_sc(sc, x̄_prior, S_δx_prior, S_δv, ỹ, h!)

    end

end

end