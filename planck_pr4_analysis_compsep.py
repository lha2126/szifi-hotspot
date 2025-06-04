## Code to search for hotspots (or other sources) using a Planck PR4 component-separated temperature map

# Imports
import numpy as np, szifi, healpy as hp, sys, os
import matplotlib.pyplot as plt


## Input options
assert len(sys.argv) == 3, "Must supply analysis type and tile number"

# Choose which type of template
# Options = 'soubhik' [inflationary], 'arnaud' [tSZ cluster], 'point' [point-source]
source_type = str(sys.argv[1])
assert source_type in ['soubhik','arnaud','point'], "Analysis type must be 'soubhik', 'arnaud', or 'point'"
# Define which cut-out to use
tile_no = int(sys.argv[2])
assert tile_no >= 0 and tile_no < 768, "Tile number must be between 0 and 767"

### DEFAULT PARAMETERS
params_szifi = szifi.params_szifi_default
params_data = szifi.params_data_default
params_model = szifi.params_model_default

# Data paths
output_dir = 'outputs_planck/' # output catalogs
params_szifi['path'] = '/mnt/home/ophilcox/szifi_hotspot/' # main code
cutout_dir = '/mnt/home/ophilcox/ceph/szifi_cutouts_new/planck_sep/' # all cutouts (will be large)
if not os.path.exists(cutout_dir): os.makedirs(cutout_dir)
if not os.path.exists(output_dir): os.makedirs(output_dir)
params_szifi['path_data'] = cutout_dir 

# Check if the output has already been computed!
if os.path.exists(output_dir+'planck_sep_batch%d_%s.npy'%(tile_no,source_type)):
    print("Output already computed; exiting!")
    sys.exit()

# Fields
params_data["field_ids"] = [tile_no]
params_data['data_set'] = 'Planck_pr4_compsep' # specifies szifi settings for the beam (experiment dependent)

# Load cosmology parameters
import camb
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.32, ombh2=0.022383, omch2=0.12011,tau=0.0543,mnu=0.06,omk=0,standard_neutrino_neff=True)
back = camb.get_background(pars)
eta_rec = back.tau_maxvis # comoving horizon at last-scattering
eta_0 = back.tau0 # comoving horizon today
chi_rec = eta_0-eta_rec # distance to comoving horizon

# Define range of template parameters
theta_500 = np.geomspace(0.5,15.,15) # theta500 (only used for tSZ)
all_eta_star = np.geomspace(10,1000,10) # eta_* in Mpc

# Define collapsed 1D grids of eta_* and eta_HS
eta_stars = []
eta_hss = []
for i in range(len(all_eta_star)):
    # Ensure that the hotspot is at the right distance
    chi_HSs = np.linspace(max(chi_rec-all_eta_star[i],0),min(chi_rec+all_eta_star[i],eta_0),12)[1:-1]
    eta_HSs = eta_0-chi_HSs
    for j in range(len(eta_HSs)):
        eta_stars.append(all_eta_star[i])
        eta_hss.append(eta_HSs[j])
eta_stars = np.asarray(eta_stars)
eta_hss = np.asarray(eta_hss)

# Load into szifi parameters
params_szifi['theta_500_vec_arcmin'] = theta_500
params_szifi['eta_star_vec'] = eta_stars
params_szifi['eta_hs_vec'] = eta_hss

# Some other options
if source_type=='arnaud':
    params_szifi['iterative'] = True # whether to iteratively mask out the sources (good if there's a signal)
    params_szifi['lrange'] = [100,2500] # ell range used in the analysis
else:
    # suggested hotspot parameters
    params_szifi['iterative'] = False
    params_szifi['lrange'] = [30,3000]

# Other parameters
params_szifi['inpaint'] = True
params_szifi['deproject_cib'] = None
params_szifi['estimate_spec'] = 'estimate'

# SNR threshold to consider a preliminary "detection"
params_szifi['q_th'] = 4.0
params_szifi['q_th_noise'] = 4.0

# Optionally save SNR maps
params_szifi['save_snr_maps'] = False
# if not os.path.exists(cutout_dir+'snr_maps/'): os.makedirs(cutout_dir+'snr_maps/')
# params_szifi['snr_maps_path'] = cutout_dir+'snr_maps/'
# params_szifi['snr_maps_name'] = 'planck_test'

# Load the frequency channels
# There's just one here, since we're using component-separated maps
params_szifi['freqs'] = [0]

# Tile parameters
nx = 1024
l = 14.8
n_tiles = 768
nside_tile = 8

### MAKE CUTOUTS
def get_cutout(inp_map, i):
    """Get cutout the i-th tile from an input full-sky map"""
    plt.ioff()
    lon,lat = hp.pix2ang(nside_tile,i,lonlat=True)
    cutout = szifi.get_cutout(inp_map,[lon,lat],nx,l)
    plt.close()
    plt.ion()
    return cutout

def get_tilemap(i):
    """Get list of non-zero pixels in the i-th tile"""
    
    smap = np.zeros(hp.nside2npix(nside_tile))
    smap[i] = 1
    umap = hp.ud_grade(smap, 2048)
    return get_cutout(umap, i)

# Test if cutouts exist
all_exist = True
for i in params_data['field_ids']:
    if not os.path.exists(cutout_dir+"planck_field_" + str(i) + "_tmap.npy"): all_exist=False
    if not os.path.exists(cutout_dir+"planck_field_" + str(i) + "_mask.npy"): all_exist=False

if all_exist:
    print("Using precomputed cutouts")
    
else:
    print("Loading Planck data")
    
    # Parameters
    freqs = ['100'] # dummy, not used directly

    # Load component-separated temperature maps
    print("Loading component-separation maps")
    freq_maps = [hp.read_map('/mnt/home/ophilcox/ceph/planck_npipe/sevem/SEVEM_NPIPE_2019/npipe6v20_sevem_cmb_005a_2048.fits')]
    
    # Load point-source mask
    print("Loading point mask")
    all_point = hp.read_map('/mnt/home/ophilcox/ceph/planck_npipe/COM_Mask_CMB-Inpainting-Mask-Int_2048_R3.00.fits')

    # Load common mask (to remove the Galactic plane)
    gal_map = hp.read_map('/mnt/home/ophilcox/ceph/planck_npipe/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits')

    # Create flat-sky cut-outs of this map.
    # These are saved and reused (i.e. they are *not* overwritten)
    # Change the "cutout_dir" if you want to make cutouts of a new map.
    print("Generating cutouts")
    for i in params_data['field_ids']:
        print("On tile %d"%i)

        # Compute cutouts of the data
        freq_cutouts = [get_cutout(freq_maps[freq_index], i) for freq_index in range(len(freqs))]
        freq_output = np.asarray([np.stack(freq_cutouts,axis=-1)])
        np.save(cutout_dir+"planck_field_" + str(i) + "_tmap.npy",freq_output)

        # Compute cutouts of the mask
        tile_cutout = get_tilemap(i)
        gal_cutout = get_cutout(gal_map, i)
        if source_type!='point':
            point_cutout = get_cutout(all_point, i)
        else:
            point_cutout = 1.+0.*gal_cutout
        mask_output = np.stack([gal_cutout, point_cutout, tile_cutout])
        
        np.save(cutout_dir+"planck_field_" + str(i) + "_mask.npy",mask_output)

# Load dataset into szifi
data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

print("\n\n## Profile: %s"%source_type)
params_model['profile_type'] = source_type

# Find all clusters (i.e. hotspots)
cluster_finder = szifi.cluster_finder(params_szifi=params_szifi,params_model=params_model,data_file=data,rank=0)
cluster_finder.find_clusters()

# Process the detections
results = cluster_finder.results_dict
detection_processor = szifi.detection_processor(results,params_szifi,params_model['profile_type'])

# Extract the catalogue of detections (noting that the tSZ clusters use iterative estimation)
if source_type=='arnaud':
    try:
        catalogue = detection_processor.results.catalogues["catalogue_find_1"]
    except KeyError:
        catalogue = szifi.cat.cluster_catalogue()
else:
    try:
        catalogue = detection_processor.results.catalogues["catalogue_find_0"]
    except KeyError:
        catalogue = szifi.cat.cluster_catalogue()

# Threshold catalogue to sources with SNR > 4.
# All catalogs will be combined later
q_th_final = 4.
try:
    catalogue = szifi.get_catalogue_q_th(catalogue,q_th_final)
except TypeError:
    catalogue = szifi.cat.cluster_catalogue()

np.save(output_dir+'planck_sep_batch%d_%s.npy'%(tile_no,source_type),catalogue)