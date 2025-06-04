import numpy as np, os, szifi, sys
import matplotlib.pyplot as plt

batch_no = int(sys.argv[1])
batch_size = 10
sim_no = 0
analysis_type = 'soubhik'
print("Using batch number %d"%batch_no)

### PARAMETERS
params_szifi = szifi.params_szifi_default
params_data = szifi.params_data_default
params_model = szifi.params_model_default

# Data paths
params_szifi['path'] = '/mnt/home/ophilcox/szifi/'
cutout_dir = '/mnt/home/ophilcox/ceph/szifi_cutouts/ffp10_sep_%d/'%sim_no
if not os.path.exists(cutout_dir): os.makedirs(cutout_dir)
params_szifi['path_data'] = cutout_dir 

# Fields
params_data["field_ids"] = np.arange(batch_no*batch_size,min([(batch_no+1)*batch_size,768]))
params_data['data_set'] = 'Planck_pr4_compsep'

import camb
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.32, ombh2=0.022383, omch2=0.12011,tau=0.0543,mnu=0.06,omk=0,standard_neutrino_neff=True)
back = camb.get_background(pars)
eta_rec = back.tau_maxvis # comoving horizon at last-scattering
eta_0 = back.tau0 # comoving horizon today
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

params_szifi['freqs'] = [0]

# Now add injection map
data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

### GENERATE COUPLING MATRICES
fac = 4

for field_id in params_data["field_ids"]:
    print("Running field ID %d"%field_id)

    mask = data.data["mask_ps"][field_id]
    pix = data.data["pix"][field_id]

    if np.array_equal(mask, szifi.maps.get_apodised_mask(data.pix,np.ones((data.nx,data.nx)),
            apotype="Smooth",aposcale=0.2)):

        if not os.path.exists(params_szifi["path_data"]+"/../apod_comp_smooth_1024.fits"):
            print("Generating smooth coupling")
            coupling_name = params_szifi["path_data"]+"/../apod_comp_smooth_1024.fits"
            ps = szifi.power_spectrum(pix,mask=mask,cm_compute=True,cm_compute_scratch=True,
                                      bin_fac=fac,cm_save=True,cm_name=coupling_name)
            print("Coupling saved to %s"%coupling_name)
        else:
            print("Smooth coupling already computed!")
    else:
        coupling_name = params_szifi["path_data"]+"/../apod_comp_smooth_" + str(field_id) + ".fits"
        if not os.path.exists(coupling_name):
            print("Generating non-smooth coupling!")
            ps = szifi.power_spectrum(pix,mask=mask,cm_compute=True,cm_compute_scratch=True,
                                      bin_fac=fac,cm_save=True,cm_name=coupling_name)
            print("Coupling saved to %s"%coupling_name)
        else:
            print("Non-smooth coupling already computed!")