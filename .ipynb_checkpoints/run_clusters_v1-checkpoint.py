import numpy as np, os, szifi, healpy as hp, sys, os
import matplotlib.pyplot as plt

sim_no = 0

batch_no = int(sys.argv[1])
batch_size = 10
assert batch_no <= (768//batch_size+1)*batch_size
print("On batch number %d"%batch_no)

### PARAMETERS
params_szifi = szifi.params_szifi_default
params_data = szifi.params_data_default
params_model = szifi.params_model_default

# Data paths
params_szifi['path'] = '/mnt/home/ophilcox/szifi/'
cutout_dir = '/mnt/home/ophilcox/ceph/szifi_cutouts/ffp10_%d/'%sim_no
if not os.path.exists(cutout_dir): os.makedirs(cutout_dir)

# Fields
params_data["field_ids"] = np.arange(batch_no*batch_size,min([(batch_no+1)*batch_size,768]))
params_data['data_set'] = 'Planck_pr4'

import camb
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.32, ombh2=0.022383, omch2=0.12011,tau=0.0543,mnu=0.06,omk=0,standard_neutrino_neff=True)
back = camb.get_background(pars)
eta_rec = back.tau_maxvis # comoving horizon at last-scattering
eta_0 = back.tau0 # comoving horizon today

# Fitting range
theta_500 = np.geomspace(0.5,15.,10)
eta_stars = np.geomspace(5,500,10)
eta_hss = np.repeat(eta_rec,len(eta_stars))

params_szifi['theta_500_vec_arcmin'] = theta_500
params_szifi['eta_star_vec'] = eta_stars
params_szifi['eta_hs_vec'] = eta_hss

params_szifi['iterative'] = False
params_szifi['lrange'] = [100,2500]

# Other parameters
params_szifi['inpaint'] = True
params_szifi['deproject_cib'] = None
params_szifi['estimate_spec'] = 'estimate'

# SNR threshold
params_szifi['q_th'] = 4.0
params_szifi['q_th_noise'] = 4.0

# Optionally save SNR maps
params_szifi['save_snr_maps'] = False
# if not os.path.exists(cutout_dir+'snr_maps/'): os.makedirs(cutout_dir+'snr_maps/')
# params_szifi['snr_maps_path'] = cutout_dir+'snr_maps/'
# params_szifi['snr_maps_name'] = 'planck_test'

params_szifi['png_signal'] = True

print("Preloading dataset")
params_szifi['path_data'] = '/mnt/home/ophilcox/ceph/szifi_cutouts/'
data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)
params_szifi['path_data'] = cutout_dir

nx = data.nx
l = data.l

n_tiles = data.n_tile
nside_tile = data.nside_tile

print("Loading simulation %d"%sim_no)

# Parameters
sim_dir = '/mnt/home/ophilcox/ceph/planck_pr4_freq/'
freqs = ['100','143','217','353','545','857']

# Load dipole
dipole_map = hp.ud_grade(hp.read_map('/mnt/home/ophilcox/ceph/planck_npipe/commander/dipole_CMB_n4096_K.fits',field=[0]),2048)

# Load frequency maps
print("Loading frequency maps")
freq_maps = []
for f in range(len(freqs)):
    freq_maps.append(hp.read_map(sim_dir+'npipe_%s/npipe6v20_%s_map.fits'%(str(sim_no+200).zfill(4),freqs[f]),field=0)-dipole_map)
    
# # Load point-source mask
# print("Loading masks")
# point_map = hp.read_map(planck_dir+'HFI_Mask_PointSrc_2048_R2.00.fits',field=[0,1,2,3,4,5])
# tot_point = np.sum([1-point_map[i] for i in range(len(point_map))],axis=0)
# all_point = (tot_point==0)

# # Load galactic mask (ordering: {20,40,60,70,80,90,97,99}%)
# gal_map = hp.read_map(planck_dir+'HFI_Mask_GalPlane-apo0_2048_R2.00.fits',field=5)

# # Check total mask
# print("Total mask: %.1f%% (for raw counts; might need further cleaning)"%(100.*np.mean(((all_point!=1)+(gal_map!=1))!=1)))

### MAKE CUTOUTS
def get_cutout(inp_map, i):
    plt.ioff()
    lon,lat = hp.pix2ang(nside_tile,i,lonlat=True)
    cutout = szifi.get_cutout(inp_map,[lon,lat],nx,l)
    plt.close()
    plt.ion()
    return cutout

def get_tilemap(i):
    """Compute tiling map for a given pixel center"""
    
    smap = np.zeros(hp.nside2npix(nside_tile))
    smap[i] = 1
    umap = hp.ud_grade(smap, 2048)
    return get_cutout(umap, i)

print("Generating cutouts")
for i in params_data['field_ids']:
    print("On tile %d"%i)

    # Compute frequency cutouts
    freq_cutouts = [get_cutout(freq_maps[freq_index], i) for freq_index in range(len(freqs))]
    freq_output = np.asarray([np.stack(freq_cutouts,axis=-1)])
    np.save(cutout_dir+"planck_field_" + str(i) + "_tmap.npy",freq_output)
    
#     # Compute mask cutouts
#     tile_cutout = get_tilemap(i)
#     gal_cutout = get_cutout(gal_map, i)
#     point_cutout = get_cutout(all_point, i)
#     mask_output = np.stack([gal_cutout, point_cutout, tile_cutout])
#     np.save(cutout_dir+"planck_field_" + str(i) + "_mask.npy",mask_output)

profiles = ['point','arnaud','soubhik']
for profile in profiles:
    print("\n\n## Profile: %s"%profile)
    params_model['profile_type'] = profile

    cluster_finder = szifi.cluster_finder(params_szifi=params_szifi,params_model=params_model,data_file=data,rank=0)
    cluster_finder.find_clusters()

    results = cluster_finder.results_dict

    detection_processor = szifi.detection_processor(results,params_szifi,params_model['profile_type'])

    catalogue_obs_noit = detection_processor.results.catalogues["catalogue_find_0"]
    #catalogue_obs_it = detection_processor.results.catalogues["catalogue_find_1"]

    #Postprocess detections

    #Reimpose threshold

    q_th_final = 5.

    catalogue_obs_noit = szifi.get_catalogue_q_th(catalogue_obs_noit,q_th_final)
    #catalogue_obs_it = szifi.get_catalogue_q_th(catalogue_obs_it,q_th_final)

    np.save('ffp10_%d_batch%d_%s.npy'%(sim_no,batch_no,profile),catalogue_obs_noit)

