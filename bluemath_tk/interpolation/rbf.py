#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import fminbound
#from scipy import stats

#from sklearn.model_selection import KFold
#from sklearn.metrics import mean_squared_error
#import xarray as xr
import pandas as pd
#import matplotlib.pyplot as plt


# RBF Phi functions

def rbf_phi_gaussian(r, const):
    return np.exp(-0.5*r*r/(const*const))

def calc_rbf_coeff(ep, x, y):

    # rbf coeff calculation
    m, n = x.shape
    A = rbf_assemble(x, rbf_phi_gaussian, ep, 0)
    b = np.concatenate((y, np.zeros((m+1,)))).reshape(-1,1)
    rbfcoeff, _, _ ,_ = np.linalg.lstsq(A, b, rcond=None)  # inverse

    return rbfcoeff, A

def rbf_assemble(x, phi, const, smooth):

    dim, n = x.shape
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            r = np.linalg.norm(x[:,i]-x[:,j])
            temp = phi(r, const)
            A[i,j] = temp
            A[j,i] = temp
        A[i,i] = A[i,i] - smooth

    # polynomial part
    P = np.hstack((np.ones((n,1)), x.T))
    A = np.vstack(
        (np.hstack((A,P)), np.hstack((P.T, np.zeros((dim+1,dim+1)))))
    )

    return A


def cost_eps(ep, x, y):

    # rbf coeff and A matrix calculation
    rbf_coeff, A = calc_rbf_coeff(ep, x, y)

    # rbf cost calculation
    m, n = x.shape
    A = A[:n,:n]
    invA = np.linalg.pinv(A)
    m1, n1 = rbf_coeff.shape
    kk = y - rbf_coeff[m1-m-1]
    for i in range(m):
        kk = kk - rbf_coeff[m1-m+i] * x[i,:]
    ceps = np.dot(invA, kk) / np.diagonal(invA)
    yy = np.linalg.norm(ceps)

    return yy

def normalize(data, ix_scalar, ix_directional, minis = None, maxis = None):
    '''
    normalize data subset - norm = (val - min) / (max - min)

    Returns:
    - data_norm: normalized data
    '''

    data_norm = np.zeros(data.shape) * np.nan

    # Calculate maxs and mins 
    if minis is None or maxis is None:
        minis = []
        maxis = []
        # Scalar data
        for ix in ix_scalar:
            v = data[:, ix]
            mi = np.nanmin(v)
            ma = np.nanmax(v)
            data_norm[:, ix] = (v - mi) / (ma - mi)
            minis.append(mi)
            maxis.append(ma)

        minis = np.array(minis)
        maxis = np.array(maxis)

    # Max and mins given
    else:

        # Scalar data
        for c, ix in enumerate(ix_scalar):
            v = data[:, ix]
            mi = minis[c]
            ma = maxis[c]
            data_norm[:, ix] = (v - mi) / (ma - mi)

    # Directional data
    for ix in ix_directional:
        v = data[:, ix]
        data_norm[:, ix] = v / 180.0
        
    return data_norm, minis, maxis



class RBF:

    '''
    This class implements the Radial Bases Function Algorithm (RBF)
    '''

    def __init__(self):
        self.data = []
        self.ix_scalar = []
        self.ix_directional = []

        self.subset = []
        self.target = []
        self.ix_scalar_subset = []
        self.ix_directional_subset = []
        self.ix_scalar_target = []
        self.ix_directional_target = []


    def rbf_interpolation(self, rbf_constant, rbf_coeff, nodes, x):
    
        phi = rbf_phi_gaussian   # gaussian RBFs
        rbf_coeff = rbf_coeff.flatten()
    
        dim, n = nodes.shape
        dim_p, n_p = x.shape
    
        f = np.zeros(n_p)
        r = np.zeros(n)
    
        for i in range(n_p):
            r = np.linalg.norm(
                np.repeat([x[:,i]], n, axis=0)-nodes.T,
                axis=1
            )
            s = rbf_coeff[n] + np.sum(rbf_coeff[:n] * phi(r, rbf_constant))
    
            # linear part
            for k in range(dim):
                s = s + rbf_coeff[k+n+1] * x[k,i]
    
            f[i] = s
    
        return f

    def rbf_calibration(self):
        '''
        Radial Basis Function (Gaussian) calibration.
    
        self.subset                - subset used for fitting RBF (dim_input)
        self.ix_scalar_subset      - scalar columns indexes for subset
        self.ix_directional_subset - directional columns indexes for subset
        self.target                - target used for fitting RBF (dim_output)
        self.ix_scalar_target      - scalar columns indexes for target
        self.ix_directional_target - directional columns indexes for target
        '''

        subset = self.subset
        target = self.target
        ix_scalar_subset = self.ix_scalar_subset
        ix_directional_subset = self.ix_directional_subset
        ix_scalar_target = self.ix_scalar_target
        ix_directional_target = self.ix_directional_target
        
        
        subset_norm, mins, maxs = normalize(
            subset, ix_scalar_subset, ix_directional_subset)

        self.subset_norm = subset_norm
        self.mins = mins
        self.maxs = maxs
        
        # RBF scalar variables 
        rbf_coeffs, opt_sigmas = [], []
        output_scalar, output_dir_x, output_dir_y = [], [], []
        
        for ix in ix_scalar_target:
            print(f'Calibrating scalar {ix}')
            v = target[:,ix]
            
            # minimize RBF cost function
            t0 = time.time()  # time counter
            
            # ensure that sigma opt is in the bounds of [sigma_min, sigma_max]
            # parameters
            sigma_min, sigma_max, d_sigma = 0.001, 0.1, 0
            while d_sigma < 0.0001:
                
                opt_sigma = fminbound(cost_eps, sigma_min, sigma_max, args=(subset_norm.T, v), disp=0)
                lm_min = np.abs(opt_sigma-sigma_min)
                lm_max = np.abs(opt_sigma-sigma_max)
                
                if lm_min < 0.0001:
                    sigma_min = sigma_min - sigma_min/2
                    
                elif lm_max < 0.001:
                    sigma_max = sigma_max + sigma_max/2
                
                d_sigma = np.nanmin([lm_min, lm_max])
                
            print('\rScalar {0}: Range sigma {1:.4f}-{2:.4f} - Opt sigma {3:.4f}'.format(ix, sigma_min, sigma_max, opt_sigma))
                
            t1 = time.time()  # optimization time
            
            # calculate RBF coeff
            rbf_coeff, _ = calc_rbf_coeff(opt_sigma, subset_norm.T, v)
            output_scalar.append(np.reshape(rbf_coeff, -1))
            
            #rbf_coeffs.append(rbf_coeff)
            opt_sigmas.append(opt_sigma)
        
        # RBF directional variables
        opt_sigma_xs, opt_sigma_ys = [], []
        
        for ix in ix_directional_target:
            print(f'Calibrating directional {ix}')
            v = target[:,ix]
    
            # x and y directional variable components
            vdg = np.pi/2 - v * np.pi/180
            pos = np.where(vdg < -np.pi)[0]
            vdg[pos] = vdg[pos] + 2 * np.pi
            vdx = np.cos(vdg)
            vdy = np.sin(vdg)
    
            # minimize RBF cost function
            t0 = time.time()  # time counter

            #directional x
            
            sigma_min, sigma_max, d_sigma = 0.001, 0.1, 0
            while d_sigma < 0.0001:

                opt_sigma_x = fminbound(cost_eps, sigma_min, sigma_max, args=(subset_norm.T, vdx))
                lm_min = np.abs(opt_sigma_x-sigma_min)
                lm_max = np.abs(opt_sigma_x-sigma_max)
                
                if lm_min < 0.0001:
                    sigma_min = sigma_min - sigma_min/2
                    
                elif lm_max < 0.001:
                    sigma_max = sigma_max + sigma_max/2
                
                d_sigma = np.nanmin([lm_min, lm_max])

            print('\rDirectional x {0}: Range sigma {1:.4f}-{2:.4f} - Opt sigma {3:.4f}'.format(ix, sigma_min, sigma_max, opt_sigma_x))
            
            #directional y
            sigma_min, sigma_max, d_sigma = 0.001, 0.1, 0
            while d_sigma < 0.0001:
                
                opt_sigma_y = fminbound(cost_eps, sigma_min, sigma_max, args=(subset_norm.T, vdy))
                lm_min = np.abs(opt_sigma_y-sigma_min)
                lm_max = np.abs(opt_sigma_y-sigma_max)
                
                if lm_min < 0.0001:
                    sigma_min = sigma_min - sigma_min/2
                    
                elif lm_max < 0.001:
                    sigma_max = sigma_max + sigma_max/2
                
                d_sigma = np.nanmin([lm_min, lm_max])

            print('\rDirectional x {0}: Range sigma {1:.4f}-{2:.4f} - Opt sigma {3:.4f}'.format(ix, sigma_min, sigma_max, opt_sigma_y))
            
            t1 = time.time()  # optimization time
    
            # calculate RBF coeff
            rbf_coeff_x, _ = calc_rbf_coeff(opt_sigma_x, subset_norm.T, vdx)
            rbf_coeff_y, _ = calc_rbf_coeff(opt_sigma_y, subset_norm.T, vdy)
            output_dir_x.append(np.reshape(rbf_coeff_x, -1))
            output_dir_y.append(np.reshape(rbf_coeff_y, -1))
            
            opt_sigma_xs.append(opt_sigma_x)
            opt_sigma_ys.append(opt_sigma_y)
        
        if output_scalar != []:
            df_rbf_scalar = pd.DataFrame(np.transpose(output_scalar))
        else:
            df_rbf_scalar = pd.DataFrame()
            opt_sigmas = []
            
        if output_dir_x != []:
            df_rbf_dirx = pd.DataFrame(np.transpose(output_dir_x))
            df_rbf_diry = pd.DataFrame(np.transpose(output_dir_y))
        else:
            df_rbf_dirx = pd.DataFrame()
            df_rbf_diry = pd.DataFrame()
            opt_sigma_x, opt_sigma_y = [], []


        self.df_rbf_scalar = df_rbf_scalar
        self.opt_sigmas_scalar = opt_sigmas

        self.df_rbf_dir_x = df_rbf_dirx
        self.df_rbf_dir_y = df_rbf_diry
        self.opt_sigmas_dir_x = opt_sigma_xs
        self.opt_sigmas_dir_y = opt_sigma_ys
            

    def rbf_reconstruction(self, dataset):
        '''
        Radial Basis Function (Gaussian) interpolator.
    
        subset                - subset used for fitting RBF (dim_input)
        self.ix_scalar_subset      - scalar columns indexes for subset
        self.ix_directional_subset - directional columns indexes for subset
        self.target                - target used for fitting RBF (dim_output)
        self.ix_scalar_target      - scalar columns indexes for target
        self.ix_directional_target - directional columns indexes for target
        self.dataset - dataset used for RBF interpolation (dim_input)
        '''

        subset_norm = self.subset_norm
        
        ix_scalar_subset = self.ix_scalar_subset
        ix_directional_subset = self.ix_directional_subset
        ix_scalar_target = self.ix_scalar_target
        ix_directional_target = self.ix_directional_target

        df_rbf_scalar = self.df_rbf_scalar
        opt_sigmas_scalar = self.opt_sigmas_scalar

        df_rbf_dir_x = self.df_rbf_dir_x
        df_rbf_dir_y = self.df_rbf_dir_y
        opt_sigmas_dir_x = self.opt_sigmas_dir_x
        opt_sigmas_dir_y = self.opt_sigmas_dir_y

        mins = self.mins
        maxs = self.maxs

        # normalize dataset

        dataset_norm, _, _ = normalize(
            dataset, ix_scalar_subset, ix_directional_subset, mins, maxs)
    
        # output storage
        output = np.zeros((dataset.shape[0], len(ix_scalar_target) + len(ix_directional_target) ))
        
        # RBF scalar variables 
        for ix in ix_scalar_target:
    
            # RBF interpolation
            t2 = time.time()  # time counter
            output[:, ix] = self.rbf_interpolation(
                opt_sigmas_scalar[ix], df_rbf_scalar[ix].values, subset_norm.T, dataset_norm.T)
            t3 = time.time()  # interpolation time
            
    
        # RBF directional variables
        for ix_idx, ix in enumerate(ix_directional_target):
        #for ix in ix_directional_target:
    
            # RBF interpolation
            t2 = time.time()  # time counter
            output_x = self.rbf_interpolation(
                opt_sigmas_dir_x[ix_idx], df_rbf_dir_x[ix_idx].values, subset_norm.T, dataset_norm.T)
            output_y = self.rbf_interpolation(
                opt_sigmas_dir_y[ix_idx], df_rbf_dir_y[ix_idx].values, subset_norm.T, dataset_norm.T)
            t3 = time.time()  # interpolation time
    
            # join x and y components
            out = np.arctan2(output_y, output_x) * 180/np.pi
            out = 90 - out
            pos = np.where(out < 0)[0]
            out[pos] = out[pos] + 360
            output[:,ix] = out
    
    
        return output
        
