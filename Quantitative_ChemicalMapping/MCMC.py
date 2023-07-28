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


def align(inputmatrix, Nvalues, ax, ay, bx, by, ww, m, b, **kwargs):
    slope = m
    intercept = b
    #ax, ay, bx, by, ww, slope, intercept = P[0:Nvalues]
    ww = int(ww)
    n = Nvalues
    ax = int(ax)
    ay = int(ay)
    bx = int(bx)
    by = int(by)

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

import bilby

class GaussianLikelihood(bilby.core.likelihood.Likelihood):
    def __init__(self, xobs, yobs, xerr, yerr, function):
        """

        Parameters
        ----------
        xobs, yobs: array_like
            The data to analyse
        xerr, yerr: array_like
            The standard deviation of the noise
        function:
            The python function to fit to the data
        """
        super(GaussianLikelihood, self).__init__(dict())
        self.xobs = xobs
        self.yobs = yobs
        self.yerr = yerr
        self.xerr = xerr
        self.function = function
        self.nvals = len(self.yobs)

    def log_likelihood(self):
        variance = (self.xerr * self.parameters["m"]) ** 2 + self.yerr**2
        model_y = self.function(self.xobs, self.nvals, **self.parameters)
        residual = self.yobs - model_y

        ll = -0.5 * np.sum(residual**2 / variance + np.log(variance))

        return -0.5 * np.sum(residual**2 / variance + np.log(variance))


def MCMC(x,y, uncert, params, pmin, pmax):
    

    names   = ["ax","ay", "bx", "by", "ww", "m", "b"]
    priors = bilby.core.prior.PriorDict()
    for i in range(5):
         priors[names[i]] = bilby.prior.Uniform(minimum=pmin[i], maximum=pmax[i], name=names[i])

    for i in range(2):
        j = i+5
        priors[names[j]] = bilby.core.prior.analytical.Gaussian(params[j], 0.1*(pmax[j]-pmin[j]))

    likelihood = GaussianLikelihood(x,y,0, uncert, align)
    sampler_kwargs = dict(priors=priors,nwalkers = 20, nsteps = 10000, outdir="emcee_maybe",sampler = "emcee")

    output = bilby.run_sampler(likelihood=likelihood, label="test",**sampler_kwargs)

    return output


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