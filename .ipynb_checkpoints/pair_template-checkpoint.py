def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

sim_no = 0
test = is_interactive()
analysis_type = 'soubhik'

import numpy as np, szifi, healpy as hp, sys, os
import matplotlib.pyplot as plt

### PARAMETERS
params_szifi = szifi.params_szifi_default
params_data = szifi.params_data_default
params_model = szifi.params_model_default

# Data paths
params_szifi['path'] = '/insomnia001/home/lha2126/szifi/'
if analysis_type =='point':
    cutout_dir = '/insomnia001/home/lha2126/ceph/szifi_cutouts/ffp10_%d_raw/'%sim_no
else:
    cutout_dir = '/insomnia001/home/lha2126/ceph/szifi_cutouts/ffp10_%d_cleaned/'%sim_no
if not os.path.exists(cutout_dir): os.makedirs(cutout_dir)
params_szifi['path_data'] = cutout_dir 

# Fields
params_data["field_ids"] = [42]
params_data['data_set'] = 'Planck_pr4'

import camb
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.32, ombh2=0.022383, omch2=0.12011,tau=0.0543,mnu=0.06,omk=0,standard_neutrino_neff=True)
back = camb.get_background(pars)
eta_rec = back.tau_maxvis # comoving horizon at last-scattering
eta_0 = back.tau0 # comoving horizon today

# Fitting range
theta_500 = np.geomspace(0.5,15.,15)
eta_stars = np.geomspace(10,1000,10)
eta_hss = np.repeat(eta_rec,len(eta_stars))

if test:
    eta_stars = eta_stars#[:2]
    eta_hss = eta_hss#[:2]
    theta_500 = theta_500#[:2]

params_szifi['theta_500_vec_arcmin'] = theta_500
params_szifi['eta_star_vec'] = eta_stars
params_szifi['eta_hs_vec'] = eta_hss

if analysis_type=='arnaud':
    params_szifi['iterative'] = True
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
if test:
    params_szifi['save_snr_maps'] = True
    if not os.path.exists(cutout_dir+'snr_maps/'): os.makedirs(cutout_dir+'snr_maps/')
    params_szifi['snr_maps_path'] = cutout_dir+'snr_maps/'
    params_szifi['snr_maps_name'] = 'planck_test'
else:
    params_szifi['save_snr_maps'] = False

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
    if not os.path.exists(cutout_dir+"planck_field_" + str(i) + "_tmap.npy"): all_exist=False
    if not os.path.exists(cutout_dir+"planck_field_" + str(i) + "_mask.npy"): all_exist=False

if all_exist:
    print("Using precomputed cutouts")
    
else:
    print("Loading simulation %d"%sim_no)

    # Parameters
    sim_dir = '/insomnia001/home/lha2126/ceph/planck_pr4_freq/'
    freqs = ['100','143','217','353','545','857']

    # Load dipole
    dipole_map = hp.ud_grade(hp.read_map('/insomnia001/home/lha2126/ceph/planck_npipe/commander/dipole_CMB_n4096_K.fits',field=[0]),2048)

    # Load frequency maps
    print("Loading frequency maps")
    freq_maps = []
    for f in range(len(freqs)):
        freq_maps.append(hp.read_map(sim_dir+'%s/npipe6v20_%s_map.fits'%(str(sim_no+200).zfill(4),freqs[f]),field=0)-dipole_map)
    
    # Load point-source mask
    planck_dir = '/insomnia001/home/lha2126/ceph/planck_pr3_raw/'
    if analysis_type != 'point':
        print("Loading point mask")
        all_point = hp.ud_grade(hp.read_map(sim_dir+'%s/point_mask_snr10.fits'%str(sim_no+200).zfill(4)),2048)

    # Load galactic mask (ordering: {20,40,60,70,80,90,97,99}%)
    gal_map = hp.read_map(planck_dir+'HFI_Mask_GalPlane-apo0_2048_R2.00.fits',field=5)

    print("Generating cutouts")
    for i in params_data['field_ids']:
        print("On tile %d"%i)

        # Compute frequency cutouts
        freq_cutouts = [get_cutout(freq_maps[freq_index], i) for freq_index in range(len(freqs))]
        freq_output = np.asarray([np.stack(freq_cutouts,axis=-1)])
        np.save(cutout_dir+"planck_field_" + str(i) + "_tmap.npy",freq_output)

        # Compute mask cutouts
        tile_cutout = get_tilemap(i)
        gal_cutout = get_cutout(gal_map, i)
        if analysis_type!='point':
            point_cutout = get_cutout(all_point, i)
        else:
            point_cutout = 1.+0.*gal_cutout
        mask_output = np.stack([gal_cutout, point_cutout, tile_cutout])
        
        np.save(cutout_dir+"planck_field_" + str(i) + "_mask.npy",mask_output)
 

data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

### Define positions
n_cluster = 300
nside = 4096

# Define seed
np.random.seed(n_cluster*42+int(1e5)*sim_no+int(1e6)+1)

lon,lat = hp.pix2ang(nside, np.random.randint(0,hp.nside2npix(nside),n_cluster+100), lonlat=True)
print("Started with %d objects"%len(lon))

# Delete closely separated objects
bad_i = []
for i in range(len(lon)-1):
    if np.min(hp.rotator.angdist([lon[i],lat[i]],[lon[i+1:],lat[i+1:]]))<3.*60*np.pi/180./60.: 
        bad_i.append(i)

print("Deleting %d too-close objects"%len(bad_i))
lon,lat = np.delete(lon,bad_i), np.delete(lat, bad_i)

# Create final catalog
lon,lat = lon[:n_cluster], lat[:n_cluster]
assert len(lon)==n_cluster

# Define eta-star values
all_eta_star = np.geomspace(10,1000,10)
rand_ind = np.random.choice(np.arange(n_cluster), replace=False, size=n_cluster)
eta_ind = np.repeat(np.arange(len(all_eta_star)),n_cluster/len(all_eta_star))[rand_ind]
eta_stars = all_eta_star[eta_ind]

# Define g value
g = 1 #sig_gs[eta_ind]*np.random.uniform(5,15,n_cluster)

chi_rec = eta_0-eta_rec
    
def define_pair(i):
    chi_HS_pos2 = np.inf
        
    while (chi_HS_pos2>chi_rec+eta_stars[i])or(chi_HS_pos2<chi_rec-eta_stars[i])or(chi_HS_pos2>eta_0)or(chi_HS_pos2<0):

        # Sample position of point 1 randomly
        chi_vals = np.linspace(max(chi_rec-eta_stars[i],0),min(chi_rec+eta_stars[i],eta_0),int(1e6))
        chi_HS_pos1 = np.random.choice(chi_vals, p = chi_vals**2/np.sum(chi_vals**2))
        eta_HS_pos1 = eta_0-chi_HS_pos1
        
        # Compute Cartesian position
        pos1_xyz = chi_HS_pos1*hp.ang2vec(lon[i],lat[i],lonlat=True)
        ang1 = np.asarray([lon[i],lat[i]])
        
        # Sample a second point uniformly from distance up to eta_* from the first
        dis12 = np.inf
        while dis12>eta_stars[i]:
            vec12 = np.random.uniform(-eta_stars[i],eta_stars[i],size=3)
            dis12 = np.sqrt(np.sum(vec12**2,axis=0))

        # Check position of point 2 from LSS. dis = eta_0-eta_HS
        pos2_xyz = pos1_xyz + vec12
        chi_HS_pos2 = np.sqrt(np.sum(pos2_xyz**2,axis=0))
        eta_HS_pos2 = eta_0-chi_HS_pos2
        ang2 = np.asarray(hp.vec2ang(pos2_xyz,lonlat=True)).ravel()
        
    return ang1, ang2, eta_HS_pos1, eta_HS_pos2

xymap = np.meshgrid(np.arange(-data.nx/2,data.nx/2)*data.dx,np.arange(-data.nx/2,data.nx/2)*data.dx)
dismap = np.sqrt(xymap[0]**2+xymap[1]**2)
inds = np.argsort(dismap.ravel())[::-1]   
from scipy.interpolate import interp1d
    
# print("Generating all templates")
# tem_interps = []
# max_dis = []
# for i in range(len(all_eta_star)):
#     print("On eta_star = %.2f"%all_eta_star[i]) 
#     this_eta_star = all_eta_star[i]
#     ids = []
#     chi_HSs = np.linspace(max(chi_rec-this_eta_star,0),min(chi_rec+this_eta_star,eta_0),12)[1:-1]
#     eta_HSs = eta_0-chi_HSs
#     for j in range(len(eta_HSs)):
#         print("On eta_HS = %.2f"%eta_HSs[j]) 
#         this_eta_hs = eta_HSs[j]
    
#         png_model = szifi.model.png({'eta_star':this_eta_star,'eta_hs':this_eta_hs},kmin=1e-6,kmax=1,lmax=3500,reduce_k=20,type="soubhik")

#         data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

#         t_tem = png_model.get_t_map_convolved(data.pix,data.data['experiment'], theta_misc=[0.-data.dx/2,0-data.dx/2],beam='real', get_nc=False,sed=None)
#         rel_t = np.abs(t_tem[:,:,0]/np.max(np.abs(t_tem[:,:,0]))).ravel()
#         thresh = 1e-3
#         while True:
#             try:
#                 dis_max = dismap.ravel()[inds][np.where(np.diff(rel_t[inds]<thresh))[0][0]]
#                 break
#             except IndexError:
#                 thresh *= 1.1
#         print(thresh)

#         tem_interps.append(interp1d(dismap[dismap<dis_max],np.asarray([t_tem[:,:,j][dismap<dis_max] for j in range(6)]), fill_value=0, bounds_error=False))
#         max_dis.append(dis_max)
#         print("0.1%% scale: %.3f radians"%dis_max)
# np.savez('all_profiles.npz',all_eta_star=all_eta_star,tem_interps=tem_interps,max_dis=max_dis)
dd = np.load('all_profiles.npz',allow_pickle=True)
all_eta_star = dd['all_eta_star']
tem_interps = dd['tem_interps']
max_dis = dd['max_dis']

ids = []
for i in range(len(all_eta_star)):
    this_eta_star = all_eta_star[i]
    chi_HSs = np.linspace(max(chi_rec-this_eta_star,0),min(chi_rec+this_eta_star,eta_0),12)[1:-1]
    eta_HSs = eta_0-chi_HSs
    for j in range(len(eta_HSs)):
        ids.append((i,j))
        
nside = 2048
template_map = np.zeros((hp.nside2npix(nside), 6))

params = []

for profile_id in range(len(lon)):
    if profile_id%10==0: print("Adding profile %d"%profile_id)
    
    this_eta_star = eta_stars[profile_id]
    ang1,ang2,eta_HS1,eta_HS2 = define_pair(profile_id)
    
    chi_HSs = np.linspace(max(chi_rec-this_eta_star,0),min(chi_rec+this_eta_star,eta_0),12)[1:-1]
    eta_HSs = eta_0-chi_HSs
    eta_HS_ind1 = np.argmin((eta_HS1-eta_HSs)**2)
    eta_HS_ind2 = np.argmin((eta_HS2-eta_HSs)**2)
    params.append({'eta_star':eta_stars[profile_id],'ang1':ang1,'ang2':ang2,'eta_HS_1':eta_HSs[eta_HS_ind1],'eta_HS_2':eta_HSs[eta_HS_ind2]})
    
    for ang, eta_HS_ind in zip([ang1,ang2],[eta_HS_ind1,eta_HS_ind2]):
        n_close = 0
        close_pix = hp.ang2pix(nside, ang[0], ang[1], lonlat=True)
        for i in range(500):
            close_pix = np.unique(np.concatenate([np.asarray(close_pix).ravel(),
                                                  hp.get_all_neighbours(nside, close_pix).ravel()]))
            # Check distances
            dis = hp.rotator.angdist(hp.pix2ang(nside, close_pix, lonlat=True), ang, lonlat=True)
            pid = np.argmin(np.sum((np.asarray((eta_ind[profile_id],eta_HS_ind))-np.asarray(ids))**2,axis=1))
            if n_close==np.sum(dis<max_dis[pid]): break
            n_close = np.sum(dis<max_dis[pid])
            if i==499: print("Not all neighbors found!")

        template_map[close_pix,:] += tem_interps[pid](dis).T
        
outmap = '/insomnia001/home/lha2126/ceph/szifi_cutouts/cutout300_pairs_sim%d.fits'%sim_no
if os.path.exists(outmap):
    os.remove(outmap)
hp.write_map(outmap, np.asarray(template_map.T,dtype='float64'), overwrite=True)
np.savez('injection300_pairs_sim%d.npz'%sim_no, params=params)#lon=lon, lat=lat, eta_stars=eta_stars)