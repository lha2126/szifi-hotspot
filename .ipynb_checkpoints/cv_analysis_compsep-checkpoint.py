import numpy as np, szifi, healpy as hp, sys, os
import matplotlib.pyplot as plt

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

test = is_interactive()
analysis_type = 'soubhik'

if test:
    batch_no = 420
    batch_size = 1
else:
    batch_no = int(sys.argv[2])*100+int(sys.argv[1])
    batch_size = 1
    assert batch_no <= (768//batch_size+1)*batch_size
    
# Check if already computed!
if os.path.exists('outputs_cv/cv_sep_batch%d_%s.npy'%(batch_no,analysis_type)):
    print("Output already computed; exiting!")
    sys.exit()

### PARAMETERS
params_szifi = szifi.params_szifi_default
params_data = szifi.params_data_default
params_model = szifi.params_model_default

# Data paths
params_szifi['path'] = '/insomnia001/home/lha2126/szifi/'
cutout_dir = '/insomnia001/home/lha2126/ceph/szifi_cutouts/cv_sep/'
if not os.path.exists(cutout_dir): os.makedirs(cutout_dir)
params_szifi['path_data'] = cutout_dir 

# Fields
params_data["field_ids"] = np.arange(batch_no*batch_size,min([(batch_no+1)*batch_size,768]))
params_data['data_set'] = 'cv_compsep'

import camb
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.32, ombh2=0.022383, omch2=0.12011,tau=0.0543,mnu=0.06,omk=0,standard_neutrino_neff=True)
back = camb.get_background(pars)
eta_rec = back.tau_maxvis # comoving horizon at last-scattering
eta_0 = back.tau0 # comoving horizon today
chi_rec = eta_0-eta_rec # distance to comoving horizon

# Fitting range
theta_500 = np.geomspace(0.5,15.,15)
all_eta_star = np.geomspace(10,1000,10)

eta_stars = []
eta_hss = []
for i in range(len(all_eta_star)):
    chi_HSs = np.linspace(max(chi_rec-all_eta_star[i],0),min(chi_rec+all_eta_star[i],eta_0),12)[1:-1]
    eta_HSs = eta_0-chi_HSs
    for j in range(len(eta_HSs)):
        eta_stars.append(all_eta_star[i])
        eta_hss.append(eta_HSs[j])
eta_stars = np.asarray(eta_stars)
eta_hss = np.asarray(eta_hss)
        
if test:
    eta_stars = eta_stars[:2]
    eta_hss = eta_hss[:2]
    theta_500 = theta_500[:2]

params_szifi['theta_500_vec_arcmin'] = theta_500
params_szifi['eta_star_vec'] = eta_stars
params_szifi['eta_hs_vec'] = eta_hss

if analysis_type=='arnaud':
    params_szifi['iterative'] = True
    params_szifi['lrange'] = [100,2500]
else:
    params_szifi['iterative'] = False
    params_szifi['lrange'] = [30,3000]

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

params_szifi['freqs'] = [0]

nx = 1024
l = 14.8

n_tiles = 768
nside_tile = 8

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


# Test if cutouts exist
all_exist = True
for i in params_data['field_ids']:
    if not os.path.exists(cutout_dir+"cv_field_" + str(i) + "_tmap.npy"): all_exist=False
    if not os.path.exists(cutout_dir+"cv_field_" + str(i) + "_mask.npy"): all_exist=False

if all_exist:
    print("Using precomputed cutouts")
    
else:
    print("Loading CV simulation")
    
    # Parameters
    freqs = ['100']

    # Load frequency maps
    print("Loading component-separation maps")
    freq_maps = [hp.read_map('/insomnia001/home/lha2126/ceph/planck_npipe/cv_sim.fits')]
    
    # Load point-source mask
    print("Loading point mask")
    all_point = hp.read_map('/insomnia001/home/lha2126/ceph/planck_npipe/COM_Mask_CMB-Inpainting-Mask-Int_2048_R3.00.fits')

    # Load common mask
    gal_map = hp.read_map('/insomnia001/home/lha2126/ceph/planck_npipe/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits')

    print("Generating cutouts")
    for i in params_data['field_ids']:
        print("On tile %d"%i)

        # Compute frequency cutouts
        freq_cutouts = [get_cutout(freq_maps[freq_index], i) for freq_index in range(len(freqs))]
        freq_output = np.asarray([np.stack(freq_cutouts,axis=-1)])
        np.save(cutout_dir+"cv_field_" + str(i) + "_tmap.npy",freq_output)

        # Compute mask cutouts
        tile_cutout = get_tilemap(i)
        gal_cutout = get_cutout(gal_map, i)
        if analysis_type!='point':
            point_cutout = get_cutout(all_point, i)
        else:
            point_cutout = 1.+0.*gal_cutout
        mask_output = np.stack([gal_cutout, point_cutout, tile_cutout])
        
        np.save(cutout_dir+"cv_field_" + str(i) + "_mask.npy",mask_output)

# Now define dataset
data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

print("\n\n## Profile: %s"%analysis_type)
params_model['profile_type'] = analysis_type
assert len(params_szifi['freqs'])==1

cluster_finder = szifi.cluster_finder(params_szifi=params_szifi,params_model=params_model,data_file=data,rank=0)
cluster_finder.find_clusters()

results = cluster_finder.results_dict
detection_processor = szifi.detection_processor(results,params_szifi,params_model['profile_type'])

if analysis_type=='arnaud':
    try:
        catalogue = detection_processor.results.catalogues["catalogue_find_1"]
    except KeyError:
        catalogue = szifi.cat.cluster_catalogue()
else:
    try:
        catalogue = detection_processor.results.catalogues["catalogue_find_0"]
    except KeyError:
        catalogue = szifi.cat.cluster_catalogue()

q_th_final = 4.
try:
    catalogue = szifi.get_catalogue_q_th(catalogue,q_th_final)
except TypeError:
    catalogue = szifi.cat.cluster_catalogue()

np.save('outputs_cv/cv_sep_batch%d_%s.npy'%(batch_no,analysis_type),catalogue)