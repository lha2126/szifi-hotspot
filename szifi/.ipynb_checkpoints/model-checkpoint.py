import numpy as np
import os
from scipy import integrate, interpolate
import scipy.optimize as optimize
from szifi import maps

#Delta critical always

class gnfw:

    def __init__(self,M,z_halo,cosmology,c=1.177,Delta=500.,type="arnaud",R_truncation=5.):

        #Create variables

        self.M = M*1e15
        self.c = c
        self.z_halo = z_halo
        self.cosmology = cosmology
        self.Delta = Delta
        self.type = type
        self.R_truncation = R_truncation

        #Calculate basic quantities

        const = constants()
        self.rho_c = cosmology.critical_density(self.z_halo).value*1000.*const.mpc**3/const.solar
        self.R_Delta = (3./4.*self.M/self.rho_c/self.Delta/np.pi)**(1./3.)
        self.chi = cosmology.comoving_distance(self.z_halo).value #In Mpc
        self.d_ad = self.chi/(1.+z_halo)

        self.theta_Delta = self.R_Delta/self.d_ad
        self.theta_Delta_arcmin = self.theta_Delta/np.pi*180.*60.
        self.my_params_sz = params_sz(cosmology,c=self.c,type=self.type,M=self.M,z=self.z_halo)
        self.my_params_sz.R_truncation = self.R_truncation

        const = constants()
        self.P2Y = const.mpc*100.*const.sigma_thomson_e/const.mass_e #converts pressure in units of keV cm-3 to dimensionless Compton y

        if self.type == "arnaud":

            self.Ez = self.cosmology.H(self.z_halo).value/self.cosmology.H0.value
            self.h70 = self.cosmology.H0.value/70.
            self.P_500 = 1.65e-3*self.Ez**(8./3.)*(self.M/3e14*self.h70)**(2./3.)*self.h70**2  #in units of keV cm-3
            self.Y_500 = self.P_500*self.R_Delta**3*4.*np.pi/3.*self.P2Y #characteristic Y, in units of Mpc^2

        #elif self.type == "battaglia":

        #    self.rho_c = cosmology.critical_density(self.z_halo).value*1000.*const.mpc**3/const.solar
    #          self.P_Delta = self.G*self.M*self.Delta*self.rho_c*self.cosmology.Omb/self.cosmology.Om0/(2.*self.R_Delta)


    def get_p_cal_map(self,pix,c_500,theta_misc=[0.,0.]):

        #theta_max = np.sqrt((nx*dx)**2+(ny*dy)**2)*0.5*1.1
        theta_max = self.theta_Delta*self.my_params_sz.R_truncation
        theta_vec = np.linspace(0.,theta_max,1000)
        p_cal_vec = np.zeros(len(theta_vec))

        for i in range(0,len(theta_vec)):

            x = theta_vec[i]/self.theta_Delta
            p_cal_vec[i] = self.get_p_cal_int(x)

        from .maps import rmap

        theta_map = rmap(pix).get_distance_map_wrt_centre(theta_misc)
        p_cal_map = np.interp(theta_map,theta_vec,p_cal_vec,right=0.)

        return p_cal_map

    def get_y_map(self,pix,theta_misc=[0.,0.]):

        p_cal_map = self.get_p_cal_map(pix,self.c,theta_misc)
        y_map = self.p_cal_to_y(p_cal_map)

        return y_map

    def get_y_norm(self,type="centre"):

        if type == "R_500":

            r = 1.

        elif type == "centre":

            r = 1e-8#1e-6

        p_cal_int = self.get_p_cal_int(r)

        return self.p_cal_to_y(p_cal_int)

    def get_y_at_angle(self,theta): #angle in rad

        return self.p_cal_to_y(self.get_p_cal_int(theta/self.theta_Delta))

    def get_y_map_convolved(self,pix,fwhm_arcmin,theta_misc=[0.,0.]):

        return maps.get_gaussian_convolution(self.get_y_map(pix,theta_misc=theta_misc),fwhm_arcmin,pix)

    def get_y_norm_convolved(self,pix,fwhm_arcmin,y_map=None,theta_misc=[0.,0.]):

        if y_map == None:

            y_map = self.get_y_map_convolved(pix,fwhm_arcmin,theta_misc)

        (theta_x,theta_y) = theta_misc
        x_coord = maps.rmap(pix).get_x_coord_map_wrt_centre(theta_x)[0,:]/self.theta_Delta
        y_coord = maps.rmap(pix).get_y_coord_map_wrt_centre(theta_y)[:,0]/self.theta_Delta

        return interpolate.interp2d(x_coord,y_coord,y_map)(1.,0.)


    def p_cal_to_y(self,p_cal_map):

        if self.type == "arnaud":

            prefactor = self.P_500*(self.M/3e14*self.h70)**(self.my_params_sz.alpha_p)*self.R_Delta

        elif self.type == "battaglia":

            prefactor = 2.61051e-18*self.cosmology.Ob0/self.cosmology.Om0*self.M*self.cosmology.H(self.z_halo).value**2/self.R_Delta*self.R_Delta/1e3

        return p_cal_map*prefactor*self.P2Y

    def get_Y_sph(self,x):  #returns spherically integrated Y, assuming self-similarity+alpha_p, in units of Mpc^2. x in units of R_500

        if x == 1:

            I = 0.6145

        elif x == 5:

            I = 1.1037

        return self.Y_500*(self.M/3e14*self.h70)**(self.my_params_sz.alpha_p)*I

    def get_Y_cyl(self,x):  #returns cylindrically integrated Y, assuming self-similarity+alpha_p. x in units of R_500.
                            #In units of Mpc^2, i.e., divide by d_A^2 to get true aperture integrated Y

        if x == 1:

            J = 0.7398

        elif x == 5:

            J = 1.1037

        #return self.Y_500*(self.M/3e14*self.h70)**(self.my_params_sz.alpha_p)*J

        return 2.925e-5*J/self.h70/self.d_ad**2*self.Ez**(2./3.)*(self.M/3e14*self.h70)**(5./3.+self.my_params_sz.alpha_p)


    def get_Y_aperture(self,x): #dimensionless (sterad) it is only valid for the Arnaud cosmology (H0=70,Om0=0.3), otherwise use get_Y_aperture_numerical

        return self.get_Y_cyl(x)

    def get_Y_aperture_numerical(self,x): #x in units of theta_500

        theta_max = self.theta_Delta*x
        theta_vec = np.linspace(0.,theta_max,1000)
        p_cal_vec = np.zeros(len(theta_vec))

        for i in range(0,len(theta_vec)):

            x = theta_vec[i]/self.theta_Delta
            p_cal_vec[i] = self.get_p_cal_int(x)

        y_vec = self.p_cal_to_y(p_cal_vec)

        self.y_vec = y_vec
        self.theta_vec = theta_vec

        y_vec[0] = 0.
        Y_aperture = integrate.simps(y_vec*2.*np.pi*theta_vec,theta_vec)

        return Y_aperture


    def pressure_profile(self,x):  #x is distance in units of R_500

        if x < self.my_params_sz.R_truncation:

            if self.type == "arnaud":

                p_cal = self.my_params_sz.P0/((self.my_params_sz.c_500*x)**self.my_params_sz.gamma*(1.+(self.my_params_sz.c_500*x)**self.my_params_sz.alpha)**((self.my_params_sz.beta-self.my_params_sz.gamma)/self.my_params_sz.alpha))

            elif self.type == "battaglia":

                p_cal = self.my_params_sz.P0*(x/self.my_params_sz.xc)**self.my_params_sz.gamma*(1.+(x/self.my_params_sz.xc)**self.my_params_sz.alpha)**(-self.my_params_sz.beta)

        else:

            p_cal = 0.

        return p_cal

    def get_p_cal_int(self,r): #r is distance in units of R_500

        def p_cal(z_los):

            x = np.sqrt(z_los**2+r**2)

            return self.pressure_profile(x)

        p_int = integrate.quad(p_cal,-self.my_params_sz.R_truncation,self.my_params_sz.R_truncation,
        epsabs=1.49e-08/100.,limit=100)[0]

        return p_int

    def get_t_map(self,pix,exp,theta_misc=[0.,0.],eval_type="standard",sed=None): #returns t map in units of muK

        if eval_type == "standard":

            y_map = self.get_y_map(pix,theta_misc=theta_misc)

        elif eval_type == "hankel":

            y_map = self.get_y_map_hankel(pix,theta_misc=theta_misc)

        n_freqs = exp.n_freqs

        if sed is None:

            sed = exp.tsz_sed

        elif sed is False:

            sed = np.ones(n_freqs)

        t_map = np.zeros((pix.nx,pix.ny,n_freqs))

        for i in range(0,n_freqs):

            t_map[:,:,i] = y_map*sed[i]

        return t_map

    def get_y_map_hankel(self,pix,theta_misc=[0.,0.]):

        def to_transform(theta):

            return np.vectorize(self.get_p_cal_int)(theta/self.theta_Delta)

        theta_range = [pix.dx/20.,self.theta_Delta*self.my_params_sz.R_truncation*10.]

        rht = maps.RadialFourierTransform()
        rprofs       = to_transform(rht.r)
        lprofs       = rht.real2harm(rprofs)
        ell_vec = rht.l
        rprofs     = rht.harm2real(lprofs)
        r, rprofs    = rht.unpad(rht.r, rprofs)

        theta_map = maps.rmap(pix).get_distance_map_wrt_centre(theta_misc)
        p_cal_map = np.interp(theta_map,r,rprofs,right=0.)
        y_map = self.p_cal_to_y(p_cal_map)

        return y_map

    def get_t_map_convolved_hankel(self,pix,exp,theta_misc=[0.,0.],beam_type="gaussian",get_nc=False,sed=None):

        theta_vec,t_vec_conv,t_vec = self.get_t_vec_convolved_hankel(pix,exp,beam_type=beam_type,get_nc=True,sed=sed)

        theta_map = maps.rmap(pix).get_distance_map_wrt_centre(theta_misc)

        t_map = np.zeros((pix.nx,pix.ny,exp.n_freqs))
        t_map_conv = np.zeros((pix.nx,pix.ny,exp.n_freqs))

        for i in range(0,exp.n_freqs):

            t_map_conv[:,:,i] = np.interp(theta_map,theta_vec,t_vec_conv[:,i],right=0.)
            t_map[:,:,i] = np.interp(theta_map,theta_vec,t_vec[:,i],right=0.)

        if get_nc == True:

            ret = t_map_conv,t_map

        else:

            ret = t_map_conv

        return ret

    def get_t_vec_convolved_hankel(self,pix,exp,beam_type="gaussian",get_nc=False,sed=None):

        if sed is None:

            sed = exp.tsz_sed

        elif sed is False:

            sed = np.ones(len(exp.tsz_sed))

        def to_transform(theta):

            return np.vectorize(self.get_p_cal_int)(theta/self.theta_Delta)

        theta_range = [pix.dx/10.,self.theta_Delta*self.my_params_sz.R_truncation*20.]

        rht = maps.RadialFourierTransform(rrange=theta_range)
        rprofs = to_transform(rht.r)
        lprofs = rht.real2harm(rprofs)
        ell_vec = rht.l

        r_temp, rprofs_temp = rht.unpad(rht.r,rprofs)

        t_vec = np.zeros((len(r_temp),exp.n_freqs))
        t_vec_conv = np.zeros((len(r_temp),exp.n_freqs))

        for i in range(0,exp.n_freqs):

            if beam_type == "gaussian":

                beam_fft = maps.get_bl(exp.FWHM[i],ell_vec)

            elif beam_type == "real":

                ell_beam,beam_fft = exp.get_beam(i)
                beam_fft = np.interp(ell_vec,ell_beam,beam_fft)

            rprofs_convolved = rht.harm2real(lprofs*beam_fft)
            r, rprofs_convolved = rht.unpad(rht.r,rprofs_convolved)

            rprofs = rht.harm2real(lprofs)
            r, rprofs = rht.unpad(rht.r,rprofs)

            t_vec_conv[:,i] = self.p_cal_to_y(rprofs_convolved)*sed[i]
            t_vec[:,i] = self.p_cal_to_y(rprofs)*sed[i]

        if get_nc == True:

            ret = r,t_vec_conv,t_vec

        else:

            ret = r,t_vec_conv

        return ret

    #NOTE: theta_misc is in rad and is [i,j] coord. Positive miscentering in i
    #means "upwards", and in j "leftwards" (i.e., in both cases towards the origin)

    def get_t_map_convolved(self,pix,exp,theta_misc=[0.,0.],theta_cart=None,beam="gaussian",get_nc=False,
    eval_type="standard",sed=None):

        if theta_cart != None:

            from .maps import get_theta_misc
            theta_misc = get_theta_misc(theta_cart,pix)

        if eval_type == "standard":

            tmap = self.get_t_map(pix,exp,theta_misc=theta_misc,eval_type=eval_type,sed=sed)
            from .maps import convolve_tmap_experiment
            tmap_convolved = convolve_tmap_experiment(pix,tmap,exp,beam_type=beam)

        elif eval_type == "hankel":

            tmap_convolved,tmap = self.get_t_map_convolved_hankel(pix,exp,theta_misc=theta_misc,beam_type=beam,get_nc=True,sed=sed)
            #tmap_convolved,tmap = self.get_t_map_convolved_hankel2(pix,exp,theta_misc=theta_misc,beam_type=beam,get_nc=True,sed=sed)

        if get_nc == True:

            ret = tmap_convolved,tmap

        else:

            ret = tmap_convolved

        return ret

from scipy.special import spherical_jn,sici,lpmn
            
class png:

    def __init__(self,model_params={},kmin=1e-6,kmax=1,lmax=3500,reduce_k=10,type="soubhik"):
        """Set-up, including defining theta_Delta width, relative to distance to cluster. "type" controls the type of profile."""

        # Create variables
        self.type = type
        self.kmin = kmin
        self.kmax = kmax
        self.lmax = lmax
        self.reduce_k = reduce_k
        
        # Calculate basic quantities
        # Run CAMB
        import camb
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.32, ombh2=0.022383, omch2=0.12011,tau=0.0543,mnu=0.06,omk=0,standard_neutrino_neff=True)
        As = 1e-10*np.exp(3.0448)
        pars.InitPower.set_params(ns=0.96605,r=0.,pivot_scalar=0.05,As=As)

        DeltaSq_zeta = As*2*np.pi**2 # primordial amplitude (from Planck)
        self.H2_dotphi = np.sqrt(As*4.*np.pi**2.) # H^2 / dot{phi} in inflation [if scale invariant]
        
        # Compute distance to last scattering (or peak thereof)
        back = camb.get_background(pars)
        self.eta_rec = back.tau_maxvis # comoving horizon at last-scattering
        self.eta_0 = back.tau0 # comoving horizon today
        
        # Load transfer function
        assert lmax<=3500, "Transfer function only computed up to lmax = 3500"
        transfer = np.load('camb_output3500.npz')
        k = transfer['k'][::reduce_k]
        filt_k = (k>kmin)*(k<kmax)
        self.transfer_k = k[filt_k]
        self.transfer_l = transfer['l'][:lmax-1]
        self.transfer_func = transfer['transfer'][0][:lmax-1,::reduce_k][:,filt_k]
        
        if self.type=="soubhik":
            self.fx_over_g = lambda x: self.H2_dotphi*(sici(x)[0]-np.sin(x))
            
            # Load parameters
            # eta_star - comoving horizon at time of pair production = hotspot size
            # eta_hs - distance from observer (intersecting with Hubble shell)
            self.eta_star = model_params['eta_star']
            self.eta_hs = model_params['eta_hs']
            if self.eta_hs == 'rec':
                self.eta_hs = self.eta_rec
            assert self.eta_hs>=self.eta_rec-self.eta_star, "eta_HS = %.2f, eta_rec = %.2f, eta_star = %.2f"%(self.eta_hs, self.eta_rec, self.eta_star)
            assert self.eta_hs<=self.eta_rec+self.eta_star, "eta_HS = %.2f, eta_rec = %.2f, eta_star = %.2f"%(self.eta_hs, self.eta_rec, self.eta_star)
            
            # Define maximum scale
            self.theta_max = np.max([0.1,np.sqrt(4.*np.pi)*np.asarray(self.eta_star)/self.eta_0])
            #assert self.eta_star<=1000, "Hot-spot radius must be smaller than patch size!"
            print("Maximum scale: %.3f radians"%self.theta_max)
            
        else:
            raise Exception("Type %s unknown!"%type)
        
    def get_y_map(self,pix,theta_misc=[0.,0.]):
        
        theta_vec = np.linspace(0.,self.theta_max,1000)
        
        if self.type=='soubhik':
            model_file = 'model_%s_eta(%.2f,%.2f)_l%d_k(%.2f,%.2f,%d).npy'%(self.type,self.eta_star,self.eta_hs,self.lmax,self.kmin,self.kmax,self.reduce_k)
            if os.path.exists(model_file):
                p_cal_vec = np.load(model_file)
            else:
                p_cal_vec = self.get_angular_profile(theta_vec)
                np.save(model_file, p_cal_vec)
        else:
            raise Exception("Unknown type %s supplied!"%self.type)

        from szifi.maps import rmap
        theta_map = rmap(pix).get_distance_map_wrt_centre(theta_misc)
        p_cal_map = np.interp(theta_map,theta_vec,p_cal_vec,right=0.)

        return p_cal_map
        
    def get_angular_profile(self,theta_vec):
        """Compute the hot-spot profile (without the factor of g, and in dimensionless units)"""

        print("Computing hotspot profile with n_l: %d, n_k: %d"%(len(self.transfer_l),len(self.transfer_k)))
        
        # Compute all Legendre polynomials
        all_legs = np.stack([lpmn(0,max(self.transfer_l),np.cos(theta))[0].ravel() for theta in theta_vec]).T[2:]

        # Compute all Bessel functions
        bessels = spherical_jn(self.transfer_l[:,None],self.transfer_k*(self.eta_0-self.eta_hs))
    
        # Perform k integral
        k_integ = integrate.simpson(bessels*self.transfer_func*self.fx_over_g(self.transfer_k*self.eta_star)/self.transfer_k,self.transfer_k)/(2.*np.pi**2.)
        return np.sum(((2.*self.transfer_l+1.)*k_integ)[:,None]*all_legs,axis=0)

    def get_t_map(self,pix,exp,theta_misc=[0.,0.],eval_type="standard",sed=None): #returns t map in units of muK

        y_map = self.get_y_map(pix,theta_misc=theta_misc)
        n_freqs = exp.n_freqs

        if sed is None:
            sed = exp.png_sed
        elif sed is False:
            sed = np.ones(n_freqs)

        t_map = np.zeros((pix.nx,pix.ny,n_freqs))

        for i in range(0,n_freqs):
            t_map[:,:,i] = y_map*sed[i]

        return t_map

    #NOTE: theta_misc is in rad and is [i,j] coord. Positive miscentering in i
    #means "upwards", and in j "leftwards" (i.e., in both cases towards the origin)

    def get_t_map_convolved(self,pix,exp,theta_misc=[0.,0.],theta_cart=None,beam="gaussian",get_nc=False,
    eval_type="standard",sed=None):
        
        if theta_cart != None:
            from szifi.maps import get_theta_misc
            theta_misc = get_theta_misc(theta_cart,pix)

        assert eval_type=="standard"

        tmap = self.get_t_map(pix,exp,theta_misc=theta_misc,eval_type=eval_type,sed=sed)
        from szifi.maps import convolve_tmap_experiment
        tmap_convolved = convolve_tmap_experiment(pix,tmap,exp,beam_type=beam)

        if get_nc == True:
            ret = tmap_convolved,tmap

        else:
            ret = tmap_convolved

        return ret
    
class params_sz: #For above class

    def __init__(self,cosmology,c,z=None,M=None,type="arnaud"):

        if type == "arnaud":

            self.P0 = 8.403*(cosmology.H0.value/70.)**(-1.5)
            self.c_500 = c #1.177 from Arnaud
            self.gamma = 0.3081
            self.alpha = 1.0510
            self.beta = 5.4905
            self.alpha_p = 0.12
            self.R_truncation = 5. #in units of R_500

        elif type == "battaglia":

            def param_scal_rel(M,z,A0,alpha_m,alpha_z):

                return A0*(M/1e14)**alpha_m*(1.+z)**alpha_z

            self.P0 = param_scal_rel(M,z,18.1,0.154,-0.758)
            self.xc = param_scal_rel(M,z,0.497,-0.00865,0.731)
            self.alpha = 1.
            self.beta = param_scal_rel(M,z,4.35,0.0393,0.415)
            self.gamma = -0.3
            self.R_truncation = 4. #in units of R_200

class constants:

    def __init__(self):

        self.c_light = 299792458. #in m/s
        self.G = 6.674e-11
        self.solar = 1.98855e30    #Solar mass in kg
        self.mpc = 3.08567758149137e22   #Mpc in m
        self.sigma_thomson_e = 6.6524587158e-25 #in cm2
        self.mass_e = 0.51099895e3 # in keV
        self.k_B = 1.38064852e-23 #in J/K
        self.T_CMB = 2.7255
        self.h = 6.62607004e-34 #in Js


def get_m_500(theta_500_arcmin,z,cosmology): #return M_500 in units of 1e15 solar masses

    theta_500 = theta_500_arcmin/60./180.*np.pi
    R_500 = theta_500*cosmology.comoving_distance(z).value/(1+z)
    const = constants()
    rho_c = cosmology.critical_density(z).value*1000.*const.mpc**3/const.solar
    M_500 = 500.*4.*np.pi/3.*rho_c*R_500**3/1e15

    return M_500

def get_theta_500_arcmin(M_500,z,cosmology): #return M_500 in units of 1e15 solar masses

    M_500 *= 1e15
    const = constants()
    rho_c = cosmology.critical_density(z).value*1000.*const.mpc**3/const.solar
    R_500 = (M_500/(500.*4.*np.pi/3.*rho_c))**(1./3.)
    theta_500 = R_500/(cosmology.comoving_distance(z).value/(1+z))
    theta_500_arcmin = theta_500*60.*180./np.pi

    return theta_500_arcmin


def g(x):

    return (np.log(1+x)-x/(1.+x))/x**3

#solve for mass 2

class mass_conversion:

    def __init__(self,cosmology,Delta1,crit1,Delta2,crit2,c1):

        self.cosmology = cosmology
        self.Delta1 = Delta1
        self.Delta2 = Delta2
        self.crit1 = crit1
        self.crit2 = crit2
        self.c1 = c1

    def get_ratio_delta(self,redshift):

        if self.crit1 == self.crit2:

            ratio_delta = np.ones(len(redshift))

        if self.crit1 == "mean" and self.crit2 == "critical":

            ratio_delta = self.cosmology.H(redshift)/(self.cosmology.Om0*(1.+redshift**3))/self.cosmology.H0

        if self.crit1 == "critical" and self.crit2 == "mean":

            ratio_delta = 1./(self.cosmology.H(redshift)/(self.cosmology.Om0*(1.+redshift**3))/self.cosmology.H0)

        ratio_delta = ratio_delta*self.Delta2/self.Delta1
        self.ratio_delta = ratio_delta

        return ratio_delta

    def get_c2(self,redshift):

        ratio_delta = self.get_ratio_delta(redshift)
        c2_vec = np.zeros(len(redshift))

        for i in range(0,len(redshift)):

            if i % 1000 == 0:

                print(i)

            def f2root(x):

                return g(x)-g(self.c1)*ratio_delta[i]

            c2_vec[i] = optimize.root_scalar(f2root,x0=5.,x1=3.).root

        self.c2 = c2_vec

        return c2_vec

    def get_m2(self,M1,redshift,R1=None): #M in units of 1e15 M_Sun

        if R1 is None:

            R1 =  get_R_Delta(M1,redshift,self.cosmology,self.Delta1,self.crit1)

        c2 = self.get_c2(redshift)
        R2 = c2/self.c1*R1
        M2 = self.ratio_delta*(c2/self.c1)**3*M1

        self.R2 = R2
        self.M2 = M2

        return (M2,R2)

def get_R_Delta(M_Delta,z,cosmology,Delta,crit):

    M_Delta = M_Delta*1e15
    const = constants()

    if crit == "critical":

        rho_c = cosmology.critical_density(z).value*1000.*const.mpc**3/const.solar
        R_Delta = (M_Delta/(Delta*4.*np.pi/3.*rho_c))**(1./3.)

    return R_Delta


class point_source:

    def __init__(self,experiment,beam_type="gaussian"):

        self.experiment = experiment
        self.beam_type = beam_type

    def get_t_map_convolved(self,pix):

        tem = np.zeros((pix.nx,pix.nx,self.experiment.n_freqs))

        for i in range(0,self.experiment.n_freqs):

            if self.beam_type == "gaussian":
                tem[:,:,i] = maps.eval_beam_real_map(pix,self.experiment.FWHM[i])
                
            elif self.beam_type =='real':
                beam_ell, beam_val = self.experiment.get_beam(i)
                
                # Create point-source map at center
                tmp_map = np.zeros((pix.nx,pix.nx))
                tmp_map[pix.nx//2-1:pix.nx//2+1,pix.nx//2-1:pix.nx//2+1] = 0.25
                
                # Add beam
                tem[:,:,i] = maps.get_convolution_isotropic(tmp_map, self.experiment.get_beam(i), pix)
                
            else:
                raise Exception("Unknown beam_type %s supplied!"%(self.beam_type))
                
            # Normalize to central value
            tem[:,:,i] = tem[:,:,i]/tem[pix.nx//2,pix.nx//2,i]

        return tem

class cosmological_model:

    def __init__(self,name="Planck15"):

        if name == "Planck15":

            from astropy.cosmology import Planck15

            self.cosmology = Planck15

        elif name == "Websky":

            self.Ob0 = 0.049
            self.Oc0 = 0.261
            self.Om0 = self.Ob0 + self.Oc0
            self.h      = 0.68
            self.ns     = 0.965
            self.sigma8 = 0.81
            self.cosmology = cp.FlatLambdaCDM(Om0=self.Om0,H0=self.h*100.,Ob0=self.Ob0)
            self.As = 2.079522e-09
