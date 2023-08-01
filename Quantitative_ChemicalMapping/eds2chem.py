# %% 
import numpy as np
import scipy.interpolate as interpolate


def draw_line(img, x0, y0, x1, y1, value = 1, pstep = None):
        mat = np.zeros_like(img)
        step_number  = int(abs(x0-x1) + abs(y0-y1)) #Number of steps
        if step_number < 2:
            step_number = 2
        else:
            pass
        step_size = 1.0/step_number #Increment size
        p = [] #Point array (you can return this and not modify the matrix in the last 2 lines)
        t = 0.0 #Step current increment
        for i in range(step_number):
            p.append([int(round(x1 * t + x0 * (1 - t))), int(round(y1 * t + y0 * (1 - t)))])
            t+=step_size
        
        p = np.asarray(p)
        mat[p[:,0],p[:, 1]] = value

        print(p)
        return img[mat.astype("bool")]

def draw_proj_box(img, x0, y0, x1, y1, value = 1, pstep = 5):
    v = np.asarray([x1, y1]) - np.asarray([x0,y0])
    theta = np.arctan(v[0]/v[1])
    perp = np.asarray([-v[1], v[0]])/(np.sqrt(v[1]**2+v[0]**2))
    max_val = np.max(img[~np.isnan(img)])

    mat = np.zeros_like(img)
    step_number  = int(abs(x0-x1) + abs(y0-y1)) #Number of steps
    if step_number < 2:
            step_number = 2
    else:
            pass
    step_size = 1.0/step_number #Increment size
    p = [] #Point array (you can return this and not modify the matrix in the last 2 lines)
    t = 0.0 #Step current increment
    for i in range(step_number):
        p.append([int(round(x1 * t + x0 * (1 - t))), int(round(y1 * t + y0 * (1 - t)))])
        t+=step_size

    conc = []
    #create array of all steps perpendicular to vector between two end points
    nsteps = np.linspace(-pstep, pstep, 2*pstep+1)
    for item in p:
        mat = np.zeros_like(img)
        #coords perpendicular to vector
        coords = np.asarray([item+j*perp for j in nsteps], dtype = "int64")
        #get all vals
        mat[coords[:,0], coords[:,1]] = value
        vals = img[mat.astype("bool")]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
             vals = np.asarray([-10*max_val])
        else:
             pass
        #only want mean to get rid of some noise
        conc.append(np.mean(vals))
    
    return np.asarray(conc)


def align(inputmatrix, Nvalues,theta, **kwargs):
    slope = theta[5]
    intercept = theta[6]
    #ax, ay, bx, by, ww, slope, intercept = P[0:Nvalues]
    ww = int(theta[4])
    n = Nvalues
    ax = int(theta[0])
    ay = int(theta[1])
    bx = int(theta[2])
    by = int(theta[3])

    vals = draw_proj_box(inputmatrix, ax, ay, bx, by, pstep = ww)
    steps = np.linspace(0, len(vals)-1, len(vals))
    steps = steps[~np.isnan(vals)]
    vals = vals[~np.isnan(vals)]
    if 0 in steps:
        pass
    else:
        if steps[0]-1 != 0:
            steps = np.concatenate([[0, steps[0]-1], steps])
            vals = np.concatenate([[0,0], vals])
        else:
            steps = np.concatenate([[0], steps])
            vals = np.concatenate([[0], vals])

    interpolation = interpolate.interp1d(steps, vals)
    trial_x = interpolation(np.linspace(0, len(vals)-1, n))

    return trial_x*slope + intercept


def test_ssfun(theta, data):
    xdata = data.xdata[0]
    ydata = data.ydata[0]
    # eval model
    ymodel = align(xdata, len(ydata), theta)
    # calc sos
    ss = sum((ymodel - ydata)**2)
    return ss

# import required packages
from pymcmcstat.MCMC import MCMC
import numpy as np
import mcmcplot.mcmatplot as mcpl
import matplotlib.pyplot as plt
from scipy.stats import norm
import pymcmcstat
print(pymcmcstat.__version__)
np.seterr(over='ignore');



def MCMC_run(x,y, uncert, params, pmin, pmax, steps = 1e4):
    

    names   = ["ax","ay", "bx", "by", "ww", "m", "b"]
    mcstat = MCMC()
    # Add data
    mcstat.data.add_data_set(x, y)
    # Define simulation options and model settings
    mcstat.simulation_options.define_simulation_options(
        nsimu=int(steps),
        updatesigma=True)
    mcstat.model_settings.define_model_settings(sos_function=test_ssfun)
    # Define model parameters
    for i in range(len(names)):
         mcstat.parameters.add_model_parameter(
            name=names[i],
            theta0=params[i],
            minimum = pmin[i],
            maximum = pmax[i])

    mcstat.run_simulation()
    # Extract results
    results = mcstat.simulation_results.results
    print(results)
    import pickle
    with open("mcmcstat.pkl", "wb") as f:
        pickle.dump()
    return results


    """# setting up bilby priors
priors = dict(
    m=bilby.core.prior.Uniform(-10, 0, "m"), c=bilby.core.prior.Uniform(0, 25, "c")
)
def model(x, m, c, **kwargs):
    y = m * x + c
    return y

sampler_kwargs = dict(priors=priors, sampler="emcee",ntemps = 10, nsamples=5000, printdt=5, outdir="mcmc2",)

gaussian_unknown_x = GaussianLikelihoodUncertainX(
    xobs=conf_int_mean[filter_arr],
    yobs=np.log10(tc[filter_arr]),
    xerr=err[filter_arr],
    yerr=0,
    function=model)

result_unknown_x2 = bilby.run_sampler(
    likelihood=gaussian_unknown_x,
    label="unknown_x",**sampler_kwargs)

    """