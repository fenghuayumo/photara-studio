# AI geometry priors for native cameras

3DGS training supports independent AI depth (`--splat-lambda-d`) and normal
(`--splat-lambda-n`) losses on pinhole, OpenCV fisheye and equirectangular cameras.
Both weights default to zero. 3DGUT constraints remain unchanged.

Pinhole generation retains its existing path. Native generation uses six
120-degree perspective faces (256–768 pixels per side), runs MoGe on each
supported face and blends overlapping predictions with a central-ray weight.
Depth is aligned independently to observed SfM points in each face, requiring
at least 16 robust samples. Unsupported faces provide no depth supervision;
normal-only generation does not require SfM scale support. If a requested map
has no valid pixels, generation fails explicitly.

Native depth matches the renderer: camera Z for fisheye and ray distance for
panorama. Face normals rotate into the source camera frame and are normalized
after blending. Invalid source rays and source masks exclude prior supervision;
prior validity does not mask RGB fitting. Nearest prediction sampling avoids
interpolating depth discontinuities. This does not guarantee seam-free geometry:
independent monocular predictions can still disagree across faces.

Native caches use PHPRIOR3 and record the model and all four fisheye coefficients.
Pinhole PHPRIOR1/2 reading and PHPRIOR2 writing remain supported. Progressive
resizing preserves depth units and validates the projection-specific intrinsics.
The editor no longer forces pinhole rectification when AI priors are enabled.

Validation: analytical sphere reprojection/normal rotations, cache roundtrips,
projection mismatch rejection, independent depth/normal maps, unsupported scales
and masks, CUDA/Vulkan loss oracles and finite differences through Gaussian
parameters for all three models. A three-view subset of the real panorama
capture was used for MoGe generation and 100-step CUDA/Vulkan training at
512x256. Both prior losses were nonzero; this is an integration smoke test,
not evidence of improved reconstruction quality. Real fisheye inference has
not been validated on a captured fisheye dataset.
