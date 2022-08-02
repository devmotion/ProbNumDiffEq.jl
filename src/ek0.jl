
Base.@kwdef struct EK0f{DT,IT} <: AbstractEK
    order::Int = 3
    diffusionmodel::DT = DynamicDiffusion()
    smooth::Bool = true
    initialization::IT = TaylorModeInit()
end
export EK0f

mutable struct EK0fCache{
    RType,ProjType,SolProjType,PType,PIType,EType,uType,duType,xType,AType,QType,matType,
    diffusionType,diffModelType,measType,puType,llType,CType,
} <: AbstractODEFilterCache
    # Constants
    d::Int                  # Dimension of the problem
    q::Int                  # Order of the prior
    A::AType
    Q::QType
    Ah::AType
    Qh::QType
    diffusionmodel::diffModelType
    R::RType
    Proj::ProjType
    SolProj::SolProjType
    # Also mutable
    P::PType
    PI::PIType
    E0::EType
    E1::EType
    E2::EType
    # Mutable stuff
    u::uType
    u_pred::uType
    u_filt::uType
    tmp::uType
    x::xType
    x_pred::xType
    x_filt::xType
    x_tmp::xType
    x_tmp2::xType
    measurement::measType
    m_tmp::measType
    pu_tmp::puType
    H::matType
    du::duType
    ddu::matType
    K1::matType
    K2::matType
    G1::matType
    G2::matType
    covmatcache::matType
    Smat::matType
    C_dxd::matType
    C_dxD::matType
    C_Dxd::matType
    C_DxD::matType
    C_2DxD::matType
    C_3DxD::matType
    default_diffusion::diffusionType
    local_diffusion::diffusionType
    global_diffusion::diffusionType
    err_tmp::duType
    log_likelihood::llType
    C1::CType
    C2::CType
end

function OrdinaryDiffEq.alg_cache(
    alg::EK0f, u, rate_prototype, ::Type{uEltypeNoUnits}, ::Type{uBottomEltypeNoUnits},
    ::Type{tTypeNoUnits}, uprev, uprev2, f, t, dt, reltol, p, calck, ::Val{IIP},
) where {IIP,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits}
    if u isa Number
        error("We currently don't support scalar-valued problems")
    end
    # @info "alg_cache EK0f"

    is_secondorder_ode = f isa DynamicalODEFunction

    q = alg.order
    d = is_secondorder_ode ? length(u[1, :]) : length(u)
    D = d * (q + 1)

    u_vec = u[:]
    t0 = t

    uType = typeof(u)
    # uElType = eltype(u_vec)
    uElType = uBottomEltypeNoUnits
    matType = Matrix{uElType}

    # Projections
    Proj = projection(1, q, uElType)
    E0, E1, E2 = Proj(0), Proj(1), Proj(2)
    @assert f isa AbstractODEFunction
    SolProj = f isa DynamicalODEFunction ? [Proj(1); Proj(0)] : Proj(0)

    # Prior dynamics
    P, PI = init_preconditioner(1, q, uElType)
    A, Q = ibm(1, q, uElType)

    initial_variance = ones(uElType, q + 1)
    x0 = Gaussian(zeros(uElType, d, q + 1), PSDMatrix(diagm(sqrt.(initial_variance))))

    # Measurement model
    R = zeros(uElType, d, d)

    # Pre-allocate a bunch of matrices
    h = zeros(uElType, d)
    H = f isa DynamicalODEFunction ? copy(E2) : copy(E1)
    if f.mass_matrix != I
        _matmul!(H, f.mass_matrix, E1)
    end

    du = f isa DynamicalODEFunction ? similar(u[2, :]) : similar(u)
    ddu = f isa DynamicalODEFunction ? zeros(uElType, d, 2d) : zeros(uElType, d, d)
    v = similar(h)
    S = PSDMatrix(zeros(uElType, q + 1, 1))
    measurement = Gaussian(v, S)
    pu_tmp =
        f isa DynamicalODEFunction ?
        Gaussian(zeros(uElType, 2d), PSDMatrix(zeros(uElType, D, 2d))) : copy(measurement)
    K = zeros(uElType, D, d)
    G = zeros(uElType, D, D)
    C1 = PSDMatrix(zeros(uElType, 2D, D))
    C2 = PSDMatrix(zeros(uElType, 3D, D))
    Smat = zeros(uElType, d, d)
    covmatcache = copy(G)

    C_dxd = zeros(uElType, d, d)
    C_dxD = zeros(uElType, d, D)
    C_Dxd = zeros(uElType, D, d)
    C_DxD = zeros(uElType, D, D)
    C_2DxD = zeros(uElType, 2D, D)
    C_3DxD = zeros(uElType, 3D, D)

    diffmodel = alg.diffusionmodel
    initdiff = initial_diffusion(diffmodel, d, q, uEltypeNoUnits)
    copy!(x0.Σ, apply_diffusion(x0.Σ, initdiff))

    Ah, Qh = copy(A), copy(Q)
    u_pred = copy(u)
    u_filt = copy(u)
    tmp = copy(u)
    x_pred = copy(x0)
    x_filt = copy(x0)
    x_tmp = copy(x0)
    x_tmp2 = copy(x0)
    m_tmp = copy(measurement)
    K2 = copy(K)
    G2 = copy(G)
    err_tmp = copy(du)

    return EK0fCache{
        typeof(R),typeof(Proj),typeof(SolProj),typeof(P),typeof(PI),typeof(E0),uType,
        typeof(du),typeof(x0),typeof(A),typeof(Q),matType,typeof(initdiff),
        typeof(diffmodel),typeof(measurement),typeof(pu_tmp),uEltypeNoUnits,typeof(C1),
    }(
        # Constants
        d, q, A, Q, Ah, Qh, diffmodel, R, Proj, SolProj, P, PI, E0, E1, E2,
        # Mutable stuff
        u, u_pred, u_filt, tmp, x0, x_pred, x_filt, x_tmp, x_tmp2, measurement, m_tmp,
        pu_tmp, H, du, ddu, K, K2, G, G2, covmatcache, Smat,
        C_dxd, C_dxD, C_Dxd, C_DxD, C_2DxD, C_3DxD,
        initdiff, initdiff * NaN, initdiff * NaN,
        err_tmp, zero(uEltypeNoUnits), C1, C2,
    )
end

function OrdinaryDiffEq.initialize!(integ::OrdinaryDiffEq.ODEIntegrator, cache::EK0fCache)
    # @info "initialize! EK0f"
    if integ.f isa DynamicalODEFunction &&
       !(integ.sol.prob.problem_type isa SecondOrderODEProblem)
        error(
            """
          The given problem is a `DynamicalODEProblem`, but not a `SecondOrderODEProblem`.
          This can not be handled by ProbNumDiffEq.jl right now. Please check if the
          problem can be formulated as a second order ODE. If not, please open a new
          github issue!
          """,
        )
    end

    if integ.opts.dense && !integ.alg.smooth
        error("To use `dense=true` you need to set `smooth=true`!")
    elseif !integ.opts.dense && integ.alg.smooth
        @warn "If you set dense=false for efficiency, you might also want to set smooth=false."
    end
    if !integ.opts.save_everystep && integ.alg.smooth
        error("If you do not save all values, you do not need to smooth!")
    end
    @assert integ.saveiter == 1

    integ.kshortsize = 1
    resize!(integ.k, integ.kshortsize)
    integ.k[1] = integ.u

    # Update the initial state to the known (given or computed with AD) initial values
    initial_update!(integ, cache)

    # These are necessary since the solution object is not 100% initialized by default
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x_filt, integ.saveiter, cache.x)
    OrdinaryDiffEq.copyat_or_push!(
        integ.sol.pu,
        integ.saveiter,
        _gaussian_mul!(cache.pu_tmp, cache.SolProj, cache.x),
    )
    return nothing
end

function OrdinaryDiffEq.perform_step!(integ, cache::EK0fCache, repeat_step=false)
    # @info "perform_step! EK0f"
    @unpack t, dt = integ
    @unpack d, q, SolProj = integ.cache
    @unpack x, x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache
    @unpack x_tmp, x_tmp2 = integ.cache
    @unpack A, Q, Ah, Qh = integ.cache

    @unpack P, PI = integ.cache
    make_preconditioner!(P, dt, 1, q)
    make_preconditioner_inv!(PI, dt, 1, q)

    tnew = t + dt

    # Build the correct matrices
    @. Ah .= PI.diag .* A .* P.diag'
    X_A_Xt!(Qh, Q, PI)
    # Ah = kron(I(d), Ah)
    # Qh = PSDMatrix(kron(I(d), Qh.R))

    # Predict the mean
    _matmul!(reshape(x_pred.μ, d, q + 1), reshape(x.μ, d, q + 1), Ah')
    # predict_mean!(x_pred, x, Ah)
    mul!(view(u_pred, :), SolProj, x_pred.μ)

    # Measure
    integ.f(cache.du, u_pred, integ.p, tnew)
    integ.destats.nf += 1
    z = cache.measurement.μ
    _matmul!(z, cache.H, x_pred.μ)
    z .-= cache.du[:]

    # Estimate diffusion, and (if adaptive) the local error estimate; Stop here if rejected
    cache.local_diffusion = estimate_local_diffusion(cache.diffusionmodel, integ)
    if integ.opts.adaptive
        integ.EEst = compute_scaled_error_estimate!(integ, cache)
        if integ.EEst >= one(integ.EEst)
            return
        end
    end

    # Predict the covariance, using either the local or global diffusion
    extrapolation_diff =
        isdynamic(cache.diffusionmodel) ? cache.local_diffusion : cache.default_diffusion
    predict_cov!(x_pred, x, Ah, Qh, cache.C_DxD, cache.C_2DxD, extrapolation_diff)

    # Compute measurement covariance only now; likelihood computation is currently broken
    compute_measurement_covariance!(cache)
    # cache.log_likelihood = logpdf(cache.measurement, zeros(d))
    # integ.sol.log_likelihood += integ.cache.log_likelihood

    # Update state and save the ODE solution value
    x_filt = update!(integ, x_pred)
    mul!(view(u_filt, :), SolProj, x_filt.μ)
    integ.u .= u_filt

    # Update the global diffusion MLE (if applicable)
    if !isdynamic(cache.diffusionmodel)
        cache.global_diffusion = estimate_global_diffusion(cache.diffusionmodel, integ)
    end

    # Advance the state
    copy!(integ.cache.x, integ.cache.x_filt)

    return nothing
end
