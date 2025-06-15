## SZiFi, the Sunyaev-Zeldovich iterative Finder

SZiFi (pronounced "sci-fi") is a Python implementation of the iterative multi-frequency matched filter (iMMF) galaxy cluster finding method, which is described in detail in [Zubeldia et al. (2023a)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.4766Z/abstract). It can be used to detect galaxy clusters with mm intensity maps through their thermal Sunyaev-Zeldovich (tSZ) signal. As a novel feature, it allows for foreground deprojection via a spectrally constrained MMF, or sciMMF (see [Zubeldia et al., 2023b](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.5123Z/abstract)). It can also be used for point source detection. If you use SZiFi in any of your projects, please cite both papers.

SZiFi has been tested extensively on [*Planck*](https://pla.esac.esa.int/#home) simulated data, and its performance is currently being tested in the context of the [Simons Observatory](https://simonsobservatory.org) (SO).

If you have any questions about how to use the code, please write to me at inigo.zubeldia (at) ast cam ac uk.

### Installation

Download the source code and do 
```
$ pip install -e .
```
You'll then be able to import SZiFi in Python with
```
import szifi
```
Dependencies: [astropy](https://www.astropy.org), [healpy](https://healpy.readthedocs.io/en/latest/), [pymaster](https://namaster.readthedocs.io), [scikit-learn](https://scikit-learn.org/stable/).

### Oliver's Hotspot Notes
This version has been heavily modified by Oliver Philcox and Colin Hill to add inflationary signatures as in https://arxiv.org/abs/2405.03738. The basic routines are the following:
- ```planck_pr4_analysis_compsep.py {TYPE} {INDEX}```. This will split the Planck temperature dataset into 768 flat-sky tiles (INDEX = 0-767) and search for either hotspots (TYPE = soubhik), tSZ clusters (TYPE = arnaud), or point-sources (TYPE = point) in each. Note that the first run will be slow since the code computes (expensive) coupling matrices that are saved for later use. The precise settings can be altered in the file. Since we're working with masked component-separated maps, we should only need TYPE = soubhik here. 
- ```Planck-Analysis.ipynb```. Analyze the Planck hotspot candidates computed using the python script.
- ```injected_analysis_compsep.py {g} {INDEX}```. This splits a simulation into 768 flat-sky tiles as before, and injects 300 well-separated hotspots with some value of g. It then seaerches for hotspots, as before.
- ```Injection-Test.ipynb```. Create a map of injected point sources which are used in the python script. Also analyze the results of the python script and make a bunch of plots.

### Luca's E-mode Functionality Notes
This version has been minorly modified but Luca Abu El-Haj to add in functionality for E-mode inflationary pairwise hotspots, extending the work of https://arxiv.org/abs/2405.03738. The basic changes to the above routines are as follows: 
-  ```planck_pr4_analysis_compsep.py {TYPE} {HMODE} {INDEX}```. Now takes an argument called hmode-- the mode of hotspot which can either be 'T', or 'E'. Currently the default is set to 'E', as such if one wanted to do a TSZ or point soutce, there is no additional impact from the adition of argument. 
-  Added a new "experiment" for component separated E-mode data, called Planck_pr4_compsep_E_data which uses the smica polarization beam.
-  Note that for camb precomputed transfer functions for this code the should be saved in a precorrected form, where all E Transfer function values must be multiplied by, $\sqrt{(l + 2) * (l + 1) * l * (l - 1)}$, and added a notebook with an example of my computation of this transfer function. This is of course not at all novel, but perhaps a harmless small addition. 
-  Created a new script ```planck_pr4_analysis_compsep_emodes.py```, which is nearly identical to the above, but differentiates between E, and T with the data and mask files.
-  Made analagous changes to the ```injected_analysis_compsep.py```, where now, ```injected_analysis_compsep_eitheroption.py {g} {INDEX} {hmode}``` will go through the same functionality as before but with an added E Mode option for the simulation.

