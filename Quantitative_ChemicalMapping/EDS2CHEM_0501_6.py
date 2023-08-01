import os
import numpy as np
import pandas as pd
#own code:
import MCMC_emcee as mcmc

def calc_an(CaO, Na2O):
    return ((CaO/56.0774)/(CaO/56.0774+2*Na2O/61.9789))*100

output_dir = ["./RES/"] 
for ii in range(len(output_dir)):
    if not os.path.exists(output_dir[ii]):
        os.makedirs(output_dir[ii])

path_parent = os.path.dirname(os.getcwd())
path_grandparent = os.path.dirname(path_parent)



##############################################
##############################################
plag_crysts = ['Sample501_6_r to c_plag1']
#regions are in [x0, x1, y0, y1]
regions = [[1600, 2200, 600, 1200]]
#in [x0,y0, x1,y1]
params_list = [[180., 135., 125., 180.,  5., 0., 1.]]
##############################################
##############################################


import chardet
with open("Sample501_6_oxide.txt", 'rb') as f:
    result = chardet.detect(f.read()) 

DF_NEW = pd.read_csv("Sample501_6_oxide.txt", skiprows=[i for i in range(51)], delimiter = "\t",
                          encoding = result['encoding'], index_col= ['Comment', 'DataSet/Point'],)

samplename = DF_NEW.index.levels[0]



DF_NEW["Anorthite"] = calc_an(DF_NEW["CaO"].to_numpy(), DF_NEW["Na2O"].to_numpy())


res = np.load("0501_6all_the_data.npz", allow_pickle = True)


plag, olivine, cpx, glass, spinel = res["phase_masks"]

gauss_plag_pca, ol_pca_img, cpx_pca_img, glass, spinel = res["scores"] 


for i in range(len(plag_crysts)):
    region = regions[i]
    xl_pca1 = gauss_plag_pca[region[0]:region[1], region[2]:region[3]]

    plag1 = DF_NEW.loc[plag_crysts[i]]

    #indparams_1 = [xl_pca1, plag1]

    # MCMC parameter setup -- a x and y coordinates (left point), b x and y coordinates (right point), window width, slope, intercept
    # pnames   = ["ax","ay", "bx", "by", "ww", "m", "b"]
    # change step sizes to better suit the ranges

    params1 = np.asarray(params_list[i])

    # minimum and maximum parameter values 
    parameter_delta = [30.,30.,30.,30.]
    pmin1 =   np.array([params1[0] - parameter_delta[0] ,params1[1] - parameter_delta[1],params1[2] - parameter_delta[2],
                         params1[3] - parameter_delta[3],  1., -10., 0.])
    pmax1 =   np.array([params1[0] + parameter_delta[0] ,params1[1] + parameter_delta[1],params1[2] + parameter_delta[2],
                         params1[3] + parameter_delta[3],    10., 10., 10.])

    an_1 = plag1.Anorthite.values / 100
    mc3_output1 = mcmc.MCMC_run(xl_pca1, an_1, an_1*0.01, params = params1,
                    pmin = pmin1, pmax = pmax1)
