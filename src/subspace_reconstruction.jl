export subspace_reconstruction_simulation


"""
    subspace_reconstruction_simulation(
        bruker_path,
        basis,
        iterations,
        regularization,
    )

Reconstruct undersampled diffusion MRI data using a simulated subspace basis
and the BART PICS reconstruction framework.

A phase correction is first applied across diffusion volumes using the phase 
measured at a reference voxel located arbitrarily at the center of the reconstructed 
volume. Although this approach provides a simple means of compensating for inter-volume 
phase variations, the choice and robustness of the reference location should be further 
evaluated. A more dedicated phase-estimation strategy could therefore be implemented 
in future developments. The corrected k-space data are then reconstructed in the subspace
domain using wavelet regularization.


# Arguments
- `bruker_path::AbstractString`: Path to the Bruker dataset.
- `basis::AbstractMatrix`: Simulated temporal subspace basis of size
  `(N_volumes, subspace_dim)`.
- `iterations::Integer`: Number of BART PICS iterations.
- `regularization::Real`: Wavelet regularization parameter.

# Returns
- `subspace_coefficients`: Reconstructed subspace coefficient images.
- `reconstructed_images`: Diffusion images projected back into the original
  diffusion-volume domain.
"""
function subspace_reconstruction_simulation(
    bruker_path::AbstractString,
    basis_dict::AbstractMatrix,
    iterations::Integer,
    regularization::Real,
    )

    # -------------------------------------------------------------------------
    # Load acquisition
    # -------------------------------------------------------------------------
    printstyled("\n[1/5] Loading acquisition data...\n"; color=:cyan, bold=true)

    b = BrukerFile(bruker_path);
    raw = RawAcquisitionData_DTI_CS(b);
    acq = AcquisitionData(raw, OffsetBruker = true);
    kdata = kDataCart(acq);

    # -------------------------------------------------------------------------
    # Initial reconstruction for phase estimation
    # -------------------------------------------------------------------------
    printstyled("\n[2/5] Performing initial reconstruction for phase estimation...\n"; color=:cyan, bold=true)
    sens_fully = espirit(acq);
    params2 = Dict{Symbol, Any}();
    params2[:reco] = "multiCoil";
    params2[:reconSize] = acq.encodingSize;
    params2[:senseMaps] = sens_fully;
    params2[:iterations] = 1;
    k_bart = kDataCart(acq);

    reco = reconstruction(acq, params2).data;

    # -------------------------------------------------------------------------
    # Determine volume center
    # -------------------------------------------------------------------------
    x,y,z = parse.(Int64,b["PVM_Matrix"])

    x2, y2, z2 = cld.((x, y, z), 2)
    
    # -------------------------------------------------------------------------
    # Phase correction across diffusion volumes
    # -------------------------------------------------------------------------
    printstyled("\n[3/5] Applying phase correction across volumes...\n"; color=:cyan, bold=true)
    kdata_corrected = copy(kdata)
    n_echo = parse.(Int64,b["PVM_NEchoImages"])


    kdata_corrected = copy(kdata)
    short_data = reco[:,:,:,:,1,1]
    for dir in 2:n_echo
        diff = angle(short_data[x2, y2, z2 ,dir])-angle(short_data[x2, y2, z2 ,1])
        kdata_corrected[:,:,:,:,dir,1] = kdata_corrected[:,:,:,:,dir,1].*exp(-im*diff)
    end

    # -------------------------------------------------------------------------
    # Convert k-space data to BART format
    # -------------------------------------------------------------------------
    data_under = copy(kdata_corrected);
    data_under = permutedims(data_under,[1 2 3 4 6 5]);


    # -------------------------------------------------------------------------
    # Estimate coil sensitivity maps with BART
    # -------------------------------------------------------------------------
    printstyled("\n[4/5] Estimating coil sensitivity maps with BART...\n"; color=:cyan, bold=true)

    k_bart = reshape(data_under,collect(size(data_under)[1:4])...,1,size(data_under,6));
    sens = bart(1,"ecalib -d3 -m1 -c 0.0",k_bart[:,:,:,:,1,1,1]);

    printstyled("\n[5/5] Running subspace reconstruction...\n"; color=:cyan, bold=true)
    # -------------------------------------------------------------------------
    # Subspace reconstruction
    # -------------------------------------------------------------------------
    basis = bart(1,"transpose 1 6",basis_dict);
    basis = bart(1,"transpose 0 5",basis);
    im_SUB = bart(1,"pics -d5 -S -e -i $(iterations) -R W:7:0:$(regularization)",k_bart,sens,B = basis);
    im_TE = bart(1,"fmac -s 64",basis,im_SUB);
    return im_SUB, im_TE
end


"""
    subspace_reconstruction_calibration(
        bruker_path,
        center_size,
        iterations,
        regularization,
    )

Reconstruct undersampled diffusion MRI data using a simulated subspace basis
and the BART PICS reconstruction framework.

The calibration data are extracted from the central region of k-space, whose
dimensions are specified by center_size. These low-resolution calibration data
are reconstructed across diffusion volumes and used to estimate the temporal
subspace basis through singular value decomposition (SVD).

A phase correction is first applied across diffusion volumes using the phase 
measured at a reference voxel located arbitrarily at the center of the reconstructed 
volume. Although this approach provides a simple means of compensating for inter-volume 
phase variations, the choice and robustness of the reference location should be further 
evaluated. A more dedicated phase-estimation strategy could therefore be implemented 
in future developments. The corrected k-space data are then reconstructed in the subspace
domain using wavelet regularization.


# Arguments
- `bruker_path::AbstractString`: Path to the Bruker dataset.
- `center_size::Integer`: Size of the fully sampled central k-space region retained for subspace calibration.
- `n_basis::Integer`: Number of basis vector retained for subspace reconstrucion.
- `iterations::Integer`: Number of BART PICS iterations.
- `regularization::Real`: Wavelet regularization parameter.

# Returns
- `subspace_coefficients`: Reconstructed subspace coefficient images.
- `reconstructed_images`: Diffusion images projected back into the original
  diffusion-volume domain.
"""
function subspace_reconstruction_calibration(
    bruker_path::AbstractString,
    center_size::Integer,
    n_basis::Integer,
    iterations::Integer,
    regularization::Real,
    )

    # -------------------------------------------------------------------------
    # Load acquisition
    # -------------------------------------------------------------------------
    printstyled("\n[1/6] Loading acquisition data...\n"; color=:cyan, bold=true)

    b = BrukerFile(bruker_path);
    raw = RawAcquisitionData_DTI_CS(b);
    acq = AcquisitionData(raw, OffsetBruker = true);
    kdata = kDataCart(acq);

    # -------------------------------------------------------------------------
    # Initial reconstruction for phase estimation
    # -------------------------------------------------------------------------
    printstyled("\n[2/6] Performing initial reconstruction for phase estimation...\n"; color=:cyan, bold=true)
    sens_fully = espirit(acq);
    params2 = Dict{Symbol, Any}();
    params2[:reco] = "multiCoil";
    params2[:reconSize] = acq.encodingSize;
    params2[:senseMaps] = sens_fully;
    params2[:iterations] = 1;
    k_bart = kDataCart(acq);

    reco = reconstruction(acq, params2).data;

    # -------------------------------------------------------------------------
    # Determine volume center
    # -------------------------------------------------------------------------
    x,y,z = parse.(Int64,b["PVM_Matrix"])

    x2, y2, z2 = cld.((x, y, z), 2)
    
    # -------------------------------------------------------------------------
    # Phase correction across diffusion volumes
    # -------------------------------------------------------------------------
    printstyled("\n[3/6] Applying phase correction across volumes...\n"; color=:cyan, bold=true)
    kdata_corrected = copy(kdata)
    n_echo = parse.(Int64,b["PVM_NEchoImages"])


    kdata_corrected = copy(kdata)
    short_data = reco[:,:,:,:,1,1]
    for dir in 2:n_echo
        diff = angle(short_data[x2, y2, z2 ,dir])-angle(short_data[x2, y2, z2 ,1])
        kdata_corrected[:,:,:,:,dir,1] = kdata_corrected[:,:,:,:,dir,1].*exp(-im*diff)
    end

    # -------------------------------------------------------------------------
    # Convert k-space data to BART format
    # -------------------------------------------------------------------------
    data_under = copy(kdata_corrected);
    data_under = permutedims(data_under,[1 2 3 4 6 5]);


    # -------------------------------------------------------------------------
    # Estimate coil sensitivity maps with BART
    # -------------------------------------------------------------------------
    printstyled("\n[4/6] Estimating coil sensitivity maps with BART...\n"; color=:cyan, bold=true)

    k_bart = reshape(data_under,collect(size(data_under)[1:4])...,1,size(data_under,6));
    sens = bart(1,"ecalib -d3 -m1 -c 0.0",k_bart[:,:,:,:,1,1,1]);


    # -------------------------------------------------------------------------
    # Basis creation
    # -------------------------------------------------------------------------
    printstyled("\n[5/6] Creatin subspace basis ...\n"; color=:cyan, bold=true)

    numChannel = parse.(Int,b["PVM_EncNReceivers"])
    k_lowRes = MRICoilSensitivities.crop(k_bart,(center_size,center_size,center_size,numChannel,1,n_echo))
    im_lowRes = ifftshift(ifft(fftshift(k_lowRes),(1,2,3)))
    im_lowRes_rss = sqrt.(sum(abs.(im_lowRes) .^ 2, dims = numChannel))
    calib_dict = reshape(im_lowRes_rss,center_size*center_size*center_size,n_echo)
    svd_obj = svd(calib_dict)

    basis = ComplexF32.(svd_obj.V)[:, 1:n_basis]    
    basis = bart(1,"transpose 1 6",basis)
    basis = bart(1,"transpose 0 5",basis)


    printstyled("\n[6/6] Running subspace reconstruction...\n"; color=:cyan, bold=true)
    # -------------------------------------------------------------------------
    # Subspace reconstruction
    # -------------------------------------------------------------------------
    im_SUB = bart(1,"pics -d5 -S -e -i $(iterations) -R W:7:0:$(regularization)",k_bart,sens,B = basis);
    im_TE = bart(1,"fmac -s 64",basis,im_SUB);
    return im_SUB, im_TE
end

