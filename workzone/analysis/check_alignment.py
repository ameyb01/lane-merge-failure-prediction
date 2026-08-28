import numpy as np

for d in ["data/workzone_complex", "data/workzone_complex_run2"]:
    lat = np.load(f"{d}/latents.npy",  mmap_mode="r")
    fea = np.load(f"{d}/features.npy", mmap_mode="r")
    L   = np.load(f"{d}/lengths.npy")
    print(f"\n{d}  arrays {lat.shape} {fea.shape}")

    extra_lat = extra_fea = 0
    for i, l in enumerate(L):
        l = int(l)
        if l < lat.shape[1] and np.any(lat[i, l] != 0): extra_lat += 1
        if l < fea.shape[1] and np.any(fea[i, l] != 0): extra_fea += 1
    print(f"  episodes with a nonzero LATENT row at index==length : {extra_lat}/{len(L)}")
    print(f"  episodes with a nonzero FEATURE row at index==length: {extra_fea}/{len(L)}")

    # any all-zero feature columns = a key silently missing from ep_info
    flat = np.asarray(fea).reshape(-1, fea.shape[2])
    dead = np.where(~flat.any(axis=0))[0]
    print(f"  all-zero feature columns: {dead.tolist()}")
