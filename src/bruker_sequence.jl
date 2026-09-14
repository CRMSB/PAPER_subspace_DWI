export RawAcquisitionData_DTI_CS

"""
    RawAcquisitionData_DTI_CS(b::BrukerFile)

Convert a Bruker dataset acquired with the a_DtiStandard_CS sequence into a 
`RawAcquisitionData` object compatible with the MRIReco functions.

Input : 
    - b::BrukerFile

Output :
    - raw::RawAcquisitionData
"""
function RawAcquisitionData_DTI_CS(b::BrukerFile)
    @assert occursin("PV-360",b["ACQ_sw_version"]) "Sequence is not acquired with PV360"

    T = Complex{MRIFiles.acqWordSize(b)} 
    filename = joinpath(b.path, "rawdata.job0")

    N = MRIFiles.pvmMatrix(b)
    factor_AntiAlias = MRIFiles.pvmAntiAlias(b)

    N = round.(Int,N .* factor_AntiAlias)

    if length(N) < 3
        N_ = ones(Int,3)
        N_[1:length(N)] .= N
        N = N_
    end


    sx = MRIFiles.pvmMatrix(b)[1]
    sy = MRIFiles.pvmMatrix(b)[2]
    sz = MRIFiles.pvmMatrix(b)[3]

    numChannel = parse.(Int,b["PVM_EncNReceivers"])
    numAvailableChannel = MRIFiles.pvmEncAvailReceivers(b)
    numSlices = MRIFiles.acqNumSlices(b)
    numEchos = parse(Int,b["PVM_NEchoImages"])
    phaseFactor = MRIFiles.acqPhaseFactor(b) 
    numRep = MRIFiles.acqNumRepetitions(b)
    numEncSteps = MRIFiles.acqSpatialSize1(b) 

    profileLength = N[1]*numChannel

    I = open(filename,"r") do fd
        read!(fd,Array{T,6}(undef, profileLength,
                                    numEchos,
                                    phaseFactor,
                                    numSlices,
                                    div(numEncSteps, phaseFactor),
                                    numRep))
    end


    # extract position of rawdata
    GradPhaseVect = parse.(Float32,b["GradPhaseVector"])
    GradPhaseVect=GradPhaseVect*sy/2;
    GradPhaseVect=round.(Int,(GradPhaseVect.-minimum(GradPhaseVect)));


    GradSliceVect = parse.(Float32,b["GradSliceVector"])
    GradSliceVect=GradSliceVect*sz/2;
    GradSliceVect=round.(Int,(GradSliceVect.-minimum(GradSliceVect)))


    # case fully -> only one mask is generated need to repmat it
    if b["EnableCS"] == "Fully"
        GradPhaseVect = repeat(vec(GradPhaseVect),1,numEchos)'
        GradSliceVect = repeat(vec(GradSliceVect),1,numEchos)'
    end


    GradPhaseVect = reshape(GradPhaseVect,numEchos,:)
    GradSliceVect = reshape(GradSliceVect,numEchos,:)

    # prepare profile
    objOrd = MRIFiles.acqObjOrder(b)
    objOrd = objOrd.-minimum(objOrd)
    gradMatrix = MRIFiles.acqGradMatrix(b)

    offset1 = MRIFiles.acqReadOffset(b)
    offset2 = MRIFiles.acqPhase1Offset(b)

    offset3 = ndims(b) == 2 ? MRIFiles.acqSliceOffset(b) : MRIFiles.pvmEffPhase2Offset(b)


    profiles = Profile[]
    for nR = 1:numRep
    for nEnc = 1:div(numEncSteps, phaseFactor)
        for nSl = 1:numSlices
        for nPhase = 1:phaseFactor
            for nEcho=1:numEchos
                counter = EncodingCounters(kspace_encode_step_1=GradPhaseVect[nEcho,nEnc],
                                            kspace_encode_step_2=GradSliceVect[nEcho,nEnc],
                                            average=0,
                                            slice=objOrd[nSl],
                                            contrast=nEcho-1,
                                            phase=0,
                                            repetition=nR-1,
                                            set=0,
                                            segment=0 )

                G = gradMatrix[:,:,nSl]
                read_dir = (G[1,1],G[2,1],G[3,1])
                phase_dir = (G[1,2],G[2,2],G[3,2])
                slice_dir = (G[1,3],G[2,3],G[3,3])

                # Not sure if the following is correct...
                pos = offset1[nSl]*G[:,1] +
                    offset2[nSl]*G[:,2] +
                    offset3[nSl]*G[:,3]

                position = (pos[1], pos[2], pos[3])

                head = AcquisitionHeader(number_of_samples=N[1], idx=counter,
                                        read_dir=read_dir, phase_dir=phase_dir,
                                        slice_dir=slice_dir, position=position,
                                        center_sample=div(N[1],2),
                                        available_channels = numChannel, #numAvailableChannel ?
                                        active_channels = numChannel)
                traj = Matrix{Float32}(undef,0,0)
                dat = map(T, reshape(I[:,nEcho,nPhase,nSl,nEnc,nR],:,numChannel))
                push!(profiles, Profile(head,traj,dat) )
            end
        end
        end
    end
    end

    params = MRIFiles.brukerParams(b)
    params["trajectory"] = "cartesian"

    params["encodedSize"] = N

    return RawAcquisitionData(params, profiles)
end
