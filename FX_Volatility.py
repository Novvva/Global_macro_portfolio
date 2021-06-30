#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:16:19 2021

"""

import numpy as np 
import pandas as pd
import time
from scipy.stats import norm 
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.optimize import root, newton_krylov


# Calls for options valuation and sensitivities
class option_val:
    '''
    This class calculates the value and sensitivities of call/put options
    Input = ('Call' or 'Put', Spot, strike, tau, r, q, sigma)
    '''

    def __init__(self, CallPut, S, K, tau, r, q, sigma):
        if CallPut.lower()[0] == 'c':
            self.type = 'Call'
        else:
            self.type = 'Put'
        self.S = S
        self.K = K
        self.tau = tau
        self.r = r
        self.q = q
        self.sigma = sigma
        self.dp = (np.log(S/K) + (r-q+0.5*sigma**2)*tau ) / (sigma*np.sqrt(tau))
        self.dm = (np.log(S/K) + (r-q-0.5*sigma**2)*tau ) / (sigma*np.sqrt(tau))
        
    def price(self):
        if self.type == 'Call':
            price = self.S*np.exp(-self.q*self.tau)*norm.cdf(self.dp) - self.K* np.exp(-self.r*self.tau)* norm.cdf(self.dm)
        else:
            price = self.K*np.exp(-self.r*self.tau)*norm.cdf(-self.dm) - self.S * norm.cdf(-self.dp)*np.exp(-self.q*self.tau)
        return price
    
    def delta(self):
        if self.type == 'Call':
            delta = np.exp(-self.q*self.tau)*norm.cdf(self.dp)
        else:
            delta = np.exp(-self.q*self.tau)*(norm.cdf(self.dp) - 1)
        return delta        
    
    def gamma(self):
        return np.exp(-self.q*self.tau)*norm.pdf(self.dp)/ (self.S * self.sigma * np.sqrt(self.tau))
    
    def vega(self):
        return (self.S * np.exp(-self.q*self.tau)* np.sqrt(self.tau)*norm.pdf(self.dp))/100
    
    def theta(self):
        if self.type == 'Call':
            theta = (self.q * self.S * np.exp(-self.q*self.tau)*norm.cdf(self.dp) -
                     self.r*self.K*np.exp(-self.r*self.tau)*norm.cdf(self.dm) - 
                     self.S*np.exp(-self.q*self.tau)*norm.pdf(self.dp)*self.sigma/(2*np.sqrt(self.tau)))/365
        else:
            theta = (-self.q * self.S * np.exp(-self.q*self.tau)*(1-norm.cdf(self.dp)) +
                     self.r*self.K*np.exp(-self.r*self.tau)*(1-norm.cdf(self.dm)) - 
                     self.S*np.exp(-self.q*self.tau)*norm.pdf(self.dp)*self.sigma/(2*np.sqrt(self.tau)))/365
        return theta

# Class for volatility surface calibration and interpolation
class vol_surface:
    '''
    This class calculates the volatility surface 
    Input = (Spot, forward points 1w/1m/3m, atm rr25 bf25 1w/1m/3m)
    Assumptions - linear interpolation for broken tenors
    '''
    
    def __init__(self, spot, fwdp1w, fwdp1m, fwdp3m, atm1w, rr1w, bf1w, atm1m, rr1m, bf1m, atm3m, rr3m, bf3m):
        self.spot = spot
        self.terms = np.array([7,30,91])/365
        self.forwards = spot + np.array([fwdp1w,fwdp1m,fwdp3m]) /10000
        self.atm = np.array([atm1w,atm1m,atm3m])
        self.rr = np.array([rr1w, rr1m, rr3m])
        self.bf = np.array([bf1w, bf1m, bf3m])

        # Construct the grid of strikes, use the longest tenor (3m), and delta 10 and 90 strikes 
        n = 5  #grid size
        vol_d90 = vol_parabolic(90,atm3m,rr3m,bf3m)
        vol_d10 = vol_parabolic(10,atm3m,rr3m,bf3m)
        k_d90 = get_strike(self.forwards[2], 1/4, vol_d90, 90)
        k_d10 = get_strike(self.forwards[2], 1/4, vol_d10, 10)
        self.strikes = np.arange(k_d90,k_d10,(k_d10-k_d90)/n)

        # Create a vol surface 
        surface = np.ones([3,n]) 
        # Calibrate surface 
        for j in range(3):
            for i in range(n):
                F = lambda h: calibration_aux(self.forwards[j],self.strikes[i],self.terms[j],h,\
                                              self.atm[j], self.rr[j], self.bf[j])
                y = newton_krylov(F, vol_d90)
                surface[j][i] = y
        self.surface = surface

    def get_vol(self, strike, tenor):
        # Interpolate the surface to get the volatility
        return int_bilin(self.strikes,self.terms,self.surface,strike,tenor)
                    
    def get_strike(self,delta,tenor):
        fwd_aux = int_lin(self.terms,self.forwards,tenor)
        F = lambda h : get_delta(fwd_aux, h, tenor, self.get_vol(h,tenor)) - delta/100
        y = newton_krylov(F,self.spot)
        return y

# Auxiliary functions 
# Linear interpolation
def int_lin(x,y,x1):
    n = len(x)
    if x1 <= x[0]:
        return y[0]
    elif x1 >= x[n-1]:
        return y[n-1]
    else:
        a = 1
        while x1 > x[a]:
            a+=1
        return (x1 - x[a-1])*(y[a]-y[a-1])/(x[a]-x[a-1]) + y[a-1]   

# Bilinear interpolation 
def int_bilin(x,y,z,x1,y1):
    n = len(x)
    m = len(y)
    if x1 <= x[0]:
        z_1 = z[:,0]
        return int_lin(y,z_1,y1)
    elif x1 >= x[n-1]:
        z_1 = z[:,n-1]
        return int_lin(y,z_1,y1)
    elif y1 <= y[0]:
        z_1 = z[0]
        return int_lin(x,z_1,x1)
    elif y1 >= y[m-1]:
        z_1 = z[m-1]
        return int_lin(x,z_1,x1)
    else:
        a = 1 
        b = 1
        while x1 > x[a]:
            a+=1
        while y1 > y[b]:
            b+=1
        return (1/((x[a]-x[a-1])*(y[b]-y[b-1])) * (z[b-1][a-1]*(x[a]-x1)*(y[b]-y1) +\
                z[b][a]*(x1-x[a-1])*(y1-y[b-1]) + z[b][a-1]*(x[a]-x1)*(y1-y[b-1]) +\
                z[b-1][a]*(x1-x[a-1])*(y[b]-y1)))

# Interpolate parabolic volatilty smile using desired delta
def vol_parabolic(delta,atm,rr,bf):
    return atm/100 - 2 * rr/100 * (delta/100 - 0.5) + 16 * bf/100 * (delta/100 - 0.5)**2

def get_strike(fwd, tau, vol, delta):
    return fwd * np.exp(0.5 * tau* vol**2 + norm.isf(delta/100)*vol*np.sqrt(tau))

def get_delta(Fwd, K, tau, sigma):
    return norm.cdf((np.log(Fwd/K)+0.5*tau*sigma**2)/(sigma*np.sqrt(tau)))

def calibration_aux(forward, strike, tau, vol, atm, rr, bf):
    delta_aux = get_delta(forward,strike,tau,vol)
    vol_aux = vol_parabolic(delta_aux*100,atm,rr,bf)
    return vol - vol_aux

