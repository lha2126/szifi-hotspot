## Code to search for hotspots using a contaminated component-separated map
# Note that the code is slow the first time it's run, since it computes coupling matrices that are then saved for later use
# We also compute and save each hotspot profile the first time we use it.
# By default, this uses the component-separated maps, which do not contain tSZ clusters. We also mask out any point sources and the galactic plane.
# We should set source_type = soubhik here. The other settings are only needed for multi-frequency analyses.

# Imports
import numpy as np, szifi, healpy as hp, sys, os
import matplotlib.pyplot as plt

## Input options
assert len(sys.argv) == 4, "Must supply g, tile number and hotspot mode "#Just adding an E/T argument

# Choose which type of template
source_type = 'soubhik'

# Which FFP10 simulation to work with
sim_no = 0

# Define which cut-out to use
g = float(sys.argv[1])
tile_no = int(sys.argv[2])
hmode=str(sys.argv[3])
assert g >= 0, "g must be at least 0"
assert tile_no >= 0 and tile_no < 768, "Tile number must be between 0 and 767"
assert hmode in ['T','E'], "Analysis type must be 'T' or' E'"
### DEFAULT PARAMETERS
params_szifi = szifi.params_szifi_default
params_data = szifi.params_data_default
params_model = szifi.params_model_default

# Data paths
output_dir = 'outputs_injected/' # output catalogs
params_szifi['path'] = '/insomnia001/home/lha2126/szifi-hotspot/' # main code
cutout_dir = '/insomnia001/home/lha2126/ceph/szifi_cutouts_new/ffp10_sep_%d/'%sim_no # all cutouts (will be large)
if not os.path.exists(cutout_dir): os.makedirs(cutout_dir)
if not os.path.exists(output_dir): os.makedirs(output_dir)
params_szifi['path_data'] = cutout_dir 

# Check if already computed!
if g==0:
    if os.path.exists('outputs_injected/ffp10_sep_%d_batch%d_%s.npy'%(sim_no,tile_no,source_type)):
        print("Output already computed; exiting!")
        sys.exit()
else:
    if os.path.exists('outputs_injected/ffp10_sep_%d_batch%d_%s_inj-g%.2f.npy'%(sim_no,tile_no,source_type,g)):
        print("Output already computed; exiting!")
        sys.exit()

# Fields
params_data["field_ids"] = [tile_no]
params_data['data_set'] = 'Planck_pr4_compsep_E_data' # specifies szifi settings for the beam (experiment dependent)

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
eta_stars = np.geomspace(10,1000,10) # eta_* in Mpc
eta_hss = np.repeat(eta_rec,len(eta_stars)) # fixed to eta_rec for this test

# Load into szifi parameters
params_szifi['theta_500_vec_arcmin'] = theta_500
params_szifi['eta_star_vec'] = eta_stars
params_szifi['eta_hs_vec'] = eta_hss
params_model['hmode'] = hmode #LHA addition: Just setting up the injection script so that it can take this extra argument with type='soubhik'

# Some other options
params_szifi['iterative'] = False
params_szifi['lrange'] = [30,3000]
params_szifi['inpaint'] = True
params_szifi['deproject_cib'] = None
params_szifi['estimate_spec'] = 'estimate'
params_szifi['save_coupling_matrix'] = True

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

# Create cutouts of the injection map, if they don't exist
all_exist = True
for i in params_data['field_ids']:
    if not os.path.exists(cutout_dir+"planck_field_" + str(i) + "_inj.npy"): all_exist=False

if all_exist:
    print("Using precomputed injection cutouts")
    
else:
    print("Loading injection cutout for simulation %d"%sim_no)

    # Load map of injected hotspots (with no data)
    injection_map = [hp.read_map('/insomnia001/home/lha2126/ceph/szifi_cutouts_new/cutout300_v3_sim%d_sep.fits'%sim_no)]
    
    print("Generating cutouts")
    for i in params_data['field_ids']:
        print("On tile %d"%i)
        freqs = ['100'] # dummy, not used directly

        # Compute mask cutouts
        inj_cutouts = [get_cutout(injection_map[freq_index], i) for freq_index in range(len(freqs))]
        inj_output = np.asarray([np.stack(inj_cutouts,axis=-1)])
        np.save(cutout_dir+"planck_field_" + str(i) + "_inj.npy",inj_output)
        
# Test if cutouts exist
all_exist = True
for i in params_data['field_ids']:
    if not os.path.exists(cutout_dir+"planck_field_" + str(i) + "_tmap.npy"): all_exist=False
    if not os.path.exists(cutout_dir+"planck_field_" + str(i) + "_mask.npy"): all_exist=False

if all_exist:
    print("Using precomputed cutouts")
    
else:
    print("Loading simulation %d"%sim_no)

    # Parameters
    freqs = ['100'] # dummy, not used directly

    # Load component-separated temperature maps (from a single FFP10 simulation)
    print("Loading component-separation maps")
    freq_maps = [hp.read_map('/insomnia001/home/lha2126/ceph/planck_npipe/sevem/SEVEM_NPIPE_sims/SEVEM_NPIPE_cmb_sim%s.fits'%(str(sim_no+200).zfill(4)))+hp.read_map('/insomnia001/home/lha2126/ceph/planck_npipe/sevem/SEVEM_NPIPE_sims/SEVEM_NPIPE_noise_sim%s.fits'%(str(sim_no+200).zfill(4)))]
    
    # Load point-source mask
    print("Loading point mask")
    all_point = hp.read_map('/insomnia001/home/lha2126/ceph/planck_npipe/COM_Mask_CMB-Inpainting-Mask-Pol_2048_R3.00.fits')

    # Load common mask (to remove the Galactic plane)
    gal_map = hp.read_map('/insomnia001/home/lha2126/ceph/planck_npipe/COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits')

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
        point_cutout = get_cutout(all_point, i)
        mask_output = np.stack([gal_cutout, point_cutout, tile_cutout])
        
        np.save(cutout_dir+"planck_field_" + str(i) + "_mask.npy",mask_output)
        
# Load dataset into szifi
data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

# Add the templates
for f in params_data['field_ids']:
    data.data['t_obs'][f] += g*np.load(cutout_dir+"planck_field_" + str(f) + "_inj.npy")[0]
print("Added g = %.2f templates"%g)

print("\n\n## Profile: %s"%source_type)
params_model['profile_type'] = source_type
assert len(params_szifi['freqs'])==1

# Find all clusters (i.e. hotspots)
cluster_finder = szifi.cluster_finder(params_szifi=params_szifi,params_model=params_model,data_file=data,rank=0)
cluster_finder.find_clusters()

# Process the detections
results = cluster_finder.results_dict
detection_processor = szifi.detection_processor(results,params_szifi,params_model['profile_type'])

# Extract the catalogue of detections
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

print("Detections SNR after combination", catalogue.catalogue['q_opt'])

if g==0:
    np.save(output_dir+'ffp10_sep_%d_batch%d_%s.npy'%(sim_no,tile_no,source_type),catalogue)
else:
    np.save(output_dir+'ffp10_sep_%d_batch%d_%s_inj-g%.2f.npy'%(sim_no,tile_no,source_type,g),catalogue)    
