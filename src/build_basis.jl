export diffusion_signal, rotation_matrix, rotate_basis, generate_hemisphere_bases, build_basis

"""
    diffusion_signal(b, gradients, D)

Simulate diffusion-weighted signals using the diffusion tensor model

    S(g) = exp(-b * g' * D * g)

where `g` is a diffusion gradient direction and `D` is the diffusion tensor.

# Arguments
- `b::Real`: Diffusion weighting (s/mm²).
- `gradients::AbstractMatrix`: Gradient directions arranged as a `3 x N` matrix.
- `D::AbstractMatrix`: `3 x 3` diffusion tensor.

# Returns
- A vector of simulated diffusion signals.
"""
function diffusion_signal(
    b::Float64,
    gradients::AbstractMatrix,
    D::AbstractMatrix,
    )
    n_directions = size(gradients, 2)
    signal = Vector{Float64}(undef, n_directions)

    for i in axes(gradients, 2)
        g = view(gradients, :, i)
        signal[i] = exp(-b * dot(g, D * g))
    end

    return signal
end


function rotation_matrix(axis::Vector{T}, theta::T) where T
    axis = normalize(axis)
    ux, uy, uz = axis
    
    return [ cos(theta) + ux^2 * (1 - cos(theta))    ux * uy * (1 - cos(theta)) - uz * sin(theta)   ux * uz * (1 - cos(theta)) + uy * sin(theta)
             uy * ux * (1 - cos(theta)) + uz * sin(theta)  cos(theta) + uy^2 * (1 - cos(theta))    uy * uz * (1 - cos(theta)) - ux * sin(theta)
             uz * ux * (1 - cos(theta)) - uy * sin(theta)  uz * uy * (1 - cos(theta)) + ux * sin(theta)  cos(theta) + uz^2 * (1 - cos(theta)) ]
end

function rotate_basis(v1::Vector{T}, v2::Vector{T}, v3::Vector{T}, angles::Vector{T}) where T
    rotated_bases = []
    
    for theta in angles
        R = rotation_matrix(v1, theta)
        v2_rot = R * v2
        v3_rot = R * v3
        push!(rotated_bases, (v1, v2_rot, v3_rot))
    end
    
    return rotated_bases
end


"""
    generate_hemisphere_bases(n_bases, n_angles)

Generate orthonormal bases whose principal directions are approximately
uniformly distributed over the upper hemisphere.

For each principal direction, `n_angles` rotations are generated around the
principal axis over the interval [0, ?).

# Arguments
- `n_bases::Integer`: Number of principal directions on the hemisphere.
- `n_angles::Integer`: Number of rotations around each principal direction.

# Returns
- A `n_bases x n_angles x 3 x 3` array.
  Each `3 x 3` matrix contains an orthonormal basis as columns.
"""
function generate_hemisphere_bases(
    n_bases::Integer,
    n_angles::Integer,
    )
    bases = Array{Float64}(undef, n_bases, n_angles, 3, 3)

    golden_angle = π * (3 - sqrt(5))

    for i in 1:n_bases

        # Quasi-uniform sampling of the upper hemisphere
        z = (i - 0.5) / n_bases
        azimuth = mod((i - 1) * golden_angle, 2π)

        radial = sqrt(1 - z^2)

        e1 = [
            radial * cos(azimuth),
            radial * sin(azimuth),
            z,
        ]

        # Choose a deterministic reference vector not parallel to e1
        reference = abs(e1[3]) < 0.9 ?
            [0.0, 0.0, 1.0] :
            [1.0, 0.0, 0.0]

        # Complete the orthonormal basis
        e2 = normalize(cross(reference, e1))
        e3 = cross(e1, e2)

        # Rotations around the principal direction
        for j in 1:n_angles
            θ = π * (j - 1) / n_angles

            R = rotation_matrix(e1, θ)

            bases[i, j, :, :] .= hcat(
                e1,
                R * e2,
                R * e3,
            )
        end
    end

    return bases
end





"""
    build_basis(
        bruker_path,
        n_bases,
        n_angles,
        b_value,
        lambda_min,
        lambda_step,
        lambda_max,
        subspace_dim,
    )

Build a simulated temporal subspace basis for diffusion MRI.

Diffusion tensors are generated from combinations of eigenvalues satisfying
λ₁ > λ₂ > λ₃ and from approximately uniformly distributed orientations over
the hemisphere. The corresponding diffusion-weighted signals are simulated,
and the temporal basis is obtained from the right singular vectors of the
signal dictionary.

# Arguments
- `bruker_path::AbstractString`: Path to the Bruker acquisition.
- `n_bases::Integer`: Number of principal diffusion directions.
- `n_angles::Integer`: Number of rotations around each principal direction.
- `b_value::Real`: Diffusion weighting in s/mm.
- `lambda_min::Real`: Minimum tensor eigenvalue.
- `lambda_step::Real`: Eigenvalue sampling step.
- `lambda_max::Real`: Maximum tensor eigenvalue.
- `subspace_dim::Integer`: Number of singular vectors retained in the basis.

# Returns
- `basis`: Complex-valued temporal subspace basis of size
  `(N_volumes, subspace_dim)`.
"""
function build_basis(
    bruker_path::AbstractString,
    n_bases::Integer,
    n_angles::Integer,
    b_val::Int64,
    lambda_min::Real,
    lambda_step::Real,
    lambda_max::Real,
    subspace_dim::Integer,
    )
    # -------------------------------------------------------------------------
    # Read diffusion acquisition parameters
    # -------------------------------------------------------------------------
    bruker_file = BrukerFile(bruker_path)

    gradients = parse.(Float64, bruker_file["PVM_DwDir"])
    n_directions = parse(Int, bruker_file["PVM_DwNDiffDir"])
    n_b0 = parse(Int, bruker_file["PVM_DwAoImages"])

    # -------------------------------------------------------------------------
    # Generate diffusion tensor eigenvalues
    # -------------------------------------------------------------------------
    eigenvalue_range = lambda_min:lambda_step:lambda_max

    eigenvalues = [
        (λ1, λ2, λ3)
        for λ1 in eigenvalue_range
        for λ2 in eigenvalue_range
        for λ3 in eigenvalue_range
        if λ1 >= λ2 >= λ3
    ]

    # -------------------------------------------------------------------------
    # Generate tensor orientations
    # -------------------------------------------------------------------------
    orientations = generate_hemisphere_bases(n_bases, n_angles)

    n_orientations = n_bases * n_angles

    orientations = reshape(
        orientations,
        n_orientations,
        3,
        3,
    )

    # -------------------------------------------------------------------------
    # Allocate simulated signal dictionary
    # -------------------------------------------------------------------------
    n_tensors = length(eigenvalues) * n_orientations
    n_volumes = n_b0 + n_directions

    signal_dictionary = Matrix{Float64}(undef, n_tensors, n_volumes)

    # Convert b value to match diffusivity units
    b_scaled = b_val * 1e-6

    # -------------------------------------------------------------------------
    # Simulate diffusion signals
    # -------------------------------------------------------------------------
    
    @threads for eigenvalue_idx in eachindex(eigenvalues)

        λ = eigenvalues[eigenvalue_idx]
        Λ = Diagonal(collect(λ))

        for orientation_idx in 1:n_orientations

            # Unique row in the dictionary
            tensor_idx =
                (eigenvalue_idx - 1) * n_orientations + orientation_idx

            V = @view orientations[orientation_idx, :, :]

            D = V * Λ * V'

            signal_dictionary[tensor_idx, 1:n_b0] .= 1.0

            signal_dictionary[tensor_idx, n_b0+1:end] .= diffusion_signal(
                b_scaled,
                gradients,
                D,
            )
        end
    end

    # -------------------------------------------------------------------------
    # Compute temporal subspace basis
    # -------------------------------------------------------------------------
    svd_result = svd(signal_dictionary)

    basis = ComplexF32.(svd_result.V[:, 1:subspace_dim])

    return basis
end
