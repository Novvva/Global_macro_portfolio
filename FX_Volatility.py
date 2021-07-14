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


# Class for option valuation and sensitivities
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
        if self.tau <=0 :
            self.dp = 0
            self.dm = 0
        else: 
            self.dp = (np.log(S/K) + (r-q+0.5*sigma**2)*tau ) / (sigma*np.sqrt(tau))
            self.dm = (np.log(S/K) + (r-q-0.5*sigma**2)*tau ) / (sigma*np.sqrt(tau))
        
    def price(self):
        if self.type == 'Call':
            if self.tau == 0:
                price = np.maximum(self.S - self.K,0)
            else:
                price = self.S*np.exp(-self.q*self.tau)*norm.cdf(self.dp) - self.K* np.exp(-self.r*self.tau)* norm.cdf(self.dm)
        else:
            if self.tau == 0:
                price = np.maximum(self.K - self.S,0)
            else:
                price = self.K*np.exp(-self.r*self.tau)*norm.cdf(-self.dm) - self.S * norm.cdf(-self.dp)*np.exp(-self.q*self.tau)
        return price
    
    def delta(self):
        if self.tau == 0:
            return 0
        elif self.type == 'Call':
            delta = np.exp(-self.q*self.tau)*norm.cdf(self.dp)
            return delta
        else:
            delta = np.exp(-self.q*self.tau)*(norm.cdf(self.dp) - 1)
            return delta        
    
    def gamma(self):
        if self.tau == 0:
            return 0
        else:
            return np.exp(-self.q*self.tau)*norm.pdf(self.dp)/ (self.S * self.sigma * np.sqrt(self.tau))
    
    def vega(self):
        if self.tau == 0:
            return 0
        else:
            return (self.S * np.exp(-self.q*self.tau)* np.sqrt(self.tau)*norm.pdf(self.dp))/100
    
    def theta(self):
        if self.tau == 0:
            return 0
        elif self.type == 'Call':
            theta = (self.q * self.S * np.exp(-self.q*self.tau)*norm.cdf(self.dp) -
                     self.r*self.K*np.exp(-self.r*self.tau)*norm.cdf(self.dm) - 
                     self.S*np.exp(-self.q*self.tau)*norm.pdf(self.dp)*self.sigma/(2*np.sqrt(self.tau)))/365
            return theta
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
    
    def get_forward(self,tenor):
        return int_lin(self.terms,self.forwards,tenor)

# Strategy to get series of P&L and sensitivities 
# Inputs series of: Vol_data(spot,fwds,ATM,RR,BF), Risk-Free rete, start-date, maturity date, Bid-offer spread,...
# ...,strategy (integer from 1 to 5), available USD cash for buying options
# OUTPUT: DataFrame with option's market value, strategy pnl, options sensitivies (delta, gamma, vega, theta) 
class performe:
    def __init__(self, vol_data, rfr, start_date, end_date, bid_offer_spread, strategy, initial_cash):
        self.start = pd.to_datetime(start_date)
        self.end = pd.to_datetime(end_date)
        self.vol = vol_data[(self.start<=vol_data.index)&(vol_data.index<=self.end)]
        self.length = self.vol.shape[0]
        self.dates = self.vol.index
        # Adjust maturity date to be the last data where we have vol information
        self.end = self.dates[self.length-1]
        self.tenor = (self.end - self.start).days /365
        self.rfr = rfr[(self.start<=rfr.index)&(rfr.index<=self.end)]
        self.bos = bid_offer_spread
        self.strategy = strategy
        self.cash = initial_cash
        self.id = str(vol_data.columns[0][0:6])
        # Is USD the strong currency 
        if vol_data.columns[0][0:3] == 'USD':
            self.aux = 1
        else:
            self.aux = 0 
        # Calculate strikes for the spread acording to the strategy (buy,sell)
        v0 = [self.vol.iloc[0][i] for i in range(13)]
        self.vol_0 = vol_surface(v0[0],v0[1],v0[2],v0[3],v0[4],v0[5],v0[6],v0[7],v0[8],v0[9],v0[10],\
                                   v0[11],v0[12])
        self.spot = v0[0]
        if strategy == 1: #Put spread with strikes delta_call 60 to delta_call 90 
            self.strikes = [self.vol_0.get_strike(70,self.tenor),self.vol_0.get_strike(80,self.tenor)]
            self.type = 'Put'
        elif strategy == 2: #Put spread with strikes delta_call 50 to delta_call 80
            self.strikes = [self.vol_0.get_strike(60,self.tenor),self.vol_0.get_strike(70,self.tenor)]
            self.type = 'Put'
        elif strategy == 3: # Do nothing
            self.strikes = [0,0]
        elif strategy == 4: #Call spread with strikes delta_call 50 to delta_call 20
            self.strikes = [self.vol_0.get_strike(40,self.tenor),self.vol_0.get_strike(30,self.tenor)]
            self.type = 'Call'
        elif strategy == 5: #Call spread with strikes delta_call 40 to delta_call 10
            self.strikes = [self.vol_0.get_strike(30,self.tenor),self.vol_0.get_strike(20,self.tenor)]
            self.type = 'Call'
        # Get initial interest rates (Local & foreign) (PUT SOME CONDITION FOR WHEN USD IS LOCAL????)
        rfr0 = self.rfr[0]
        fwd0 = self.vol_0.get_forward(self.tenor)
        # if USD rate is the foreign currency
        if self.aux == 1:
            rf0 = np.log(1+rfr0*self.tenor)/self.tenor
            rl0 = np.log(fwd0 * np.exp(rf0*self.tenor) / v0[0])/self.tenor
        else: # if USD rate is the local currency
            rl0 = np.log(1+rfr0*self.tenor)/self.tenor
            rf0 = np.log(v0[0] * np.exp(rl0*self.tenor)/ fwd0)/self.tenor
        # Calculate strategy price
        if strategy == 3: 
            self.premium = 0
            self.notional = 0 
        else:
            v1 = self.vol_0.get_vol(self.strikes[0],self.tenor)
            v2 = self.vol_0.get_vol(self.strikes[1],self.tenor)
            self.opt1 = option_val(self.type, v0[0], self.strikes[0], self.tenor, rl0, rf0, v1 + self.bos/100)
            self.opt2 = option_val(self.type, v0[0], self.strikes[1], self.tenor, rl0, rf0, v2) 
            self.premium = (self.opt1.price() - self.opt2.price())/ v0[0] # Premium in strong currency with 1 unit of notional
            if self.aux == 0: # Convert cash to strong currency if its not in strong currency
                self.cash = self.cash / v0[0] 
            self.notional = self.cash / self.premium # Its in strong currency 
            self.vega_init = self.opt1.vega() * self.notional / v0[0] # in strong currency
        
    def pnl(self):
        pnl = pd.DataFrame()
        pnl['DATE']=pd.to_datetime(self.dates)
        pnl.set_index('DATE',inplace=True)
        pnl['borrow_rate'] = self.rfr + .01
        pnl['spot'] = self.spot
        if self.strategy == 3:
            pnl[['mv','premium',self.id,'delta','gamma','vega','theta']] = 0
            pnl.columns = ['borrow_rate',self.id+'_spot',self.id+'_premium',self.id+'_mv',self.id+'_pnl',\
                          self.id+'_delta',self.id+'_gamma',self.id+'_vega',self.id+'_theta']
            pnl.drop(['borrow_rate'], axis=1,inplace=True)
            return pnl
        else: 
            if self.aux == 1:
                pnl['premium'] = self.cash
            else:
                pnl['premium'] = self.cash * self.spot
            pnl['mv'] = self.cash
            pnl['mv'].iloc[0] = self.cash - self.vega_init * self.bos
            if self.aux == 1:
                pnl['pnl'] = pnl['mv'] - pnl['premium']
            else: 
                pnl['pnl'] = pnl['mv'] - pnl['premium'] / self.spot
            # Create space fo the greeks 
            pnl['delta'] = (self.opt1.delta() - self.opt2.delta()) * self.notional 
            pnl['gamma'] = (self.opt1.gamma() - self.opt2.gamma()) * self.notional 
            pnl ['vega'] = (self.opt1.vega() - self.opt2.vega()) * self.notional * self.spot
            pnl ['theta'] = (self.opt1.theta() - self.opt2.theta()) * self.notional * self.spot
            for i in range(self.length-1):
                # Calculate day-count fraction for premium accrual 
                dcf = (self.dates[i+1]-self.dates[i]).days /365
                pnl['premium'].iloc[i+1] = pnl['premium'].iloc[i] * (1 + self.rfr[i]*dcf)
                new_tenor = (self.end - self.dates[i+1]).days /365
                # Calibrate new vol surface 
                v0 = [self.vol.iloc[i+1][x] for x in range(13)]
                vol = vol_surface(v0[0],v0[1],v0[2],v0[3],v0[4],v0[5],v0[6],v0[7],v0[8],v0[9],v0[10],\
                                       v0[11],v0[12])
                # Get spot rate (DELETE)
                pnl['spot'].iloc[i+1] = v0[0]
                # Get new interest rates 
                rfr = self.rfr[i+1]
                fwd = vol.get_forward(new_tenor)
                try:
                    r_f = np.log(1+ rfr * new_tenor)/new_tenor
                    r_l = np.log(fwd * np.exp(r_f * new_tenor) / v0[0])/new_tenor
                except:
                    r_f = 0
                    r_l = 0
                # Calculate value of strategy 
                opt1 = option_val(self.type, v0[0], self.strikes[0], new_tenor, r_l, r_f, vol.get_vol(self.strikes[0],new_tenor))
                opt2 = option_val(self.type, v0[0], self.strikes[1], new_tenor, r_l, r_f, vol.get_vol(self.strikes[1],new_tenor))
                pnl['mv'].iloc[i+1] = (opt1.price() - opt2.price()) * self.notional / v0[0] # Check this spot is not
                #i
                if self.aux == 1:
                    pnl['pnl'].iloc[i+1] = pnl['mv'].iloc[i+1] - pnl['premium'].iloc[i+1]
                else:
                    pnl['pnl'].iloc[i+1] = pnl['mv'].iloc[i+1] - pnl['premium'].iloc[i+1] / v0[0]
                pnl['delta'].iloc[i+1] = (opt1.delta() - opt2.delta()) * self.notional # its in strong currrency
                pnl['gamma'].iloc[i+1] = (opt1.gamma() - opt2.gamma()) * self.notional # in strong currency 
                pnl ['vega'].iloc[i+1] = (opt1.vega() - opt2.vega()) * self.notional * v0[0] # flipped to strong currency 
                pnl ['theta'].iloc[i+1] = (opt1.theta() - opt2.theta()) * self.notional * v0[0]# flipped to strong week currency
            if self.aux == 0:
                pnl['mv'] = pnl['mv'] * pnl['spot']
                pnl['pnl'] = pnl['pnl'] * pnl['spot']
                pnl['delta'] = pnl['delta'] * pnl['spot']
                pnl['gamma'] = pnl['gamma'] * pnl['spot']
                pnl['vega'] = pnl['vega'] * pnl['spot']
                pnl['theta'] = pnl['vega'] * pnl['spot']
            pnl.columns = ['borrow_rate',self.id+'_spot',self.id+'_premium',self.id+'_mv',self.id+'_pnl',\
                          self.id+'_delta',self.id+'_gamma',self.id+'_vega',self.id+'_theta']
            pnl.drop(['borrow_rate'], axis=1,inplace=True)
            return pnl
        
# Process vol_data
allbase = pd.read_csv('data/vol_data.csv')
def clean_vol_date(allbase):
    # Index in datetime format 
    allbase['DATE']=pd.to_datetime(allbase.DATE)
    allbase.set_index('DATE',inplace=True)
    # Create the currency list 
    ccy = [x[0:6] for x in allbase.columns]
    ccy = list(set(ccy))
    # create one dataframe for each currency 
    for i in range(len(ccy)):
        globals()[str(list(ccy)[i])] = pd.DataFrame()
        for j in allbase.columns:
            if j[0:6] == list(ccy)[i]:
                globals()[str(list(ccy)[i])][j] = allbase[j]
    # Create a list with all the vols DataFrames (Not sure i will use it)
    vols = []
    for i in ccy:
        vols.append(globals()[i])
    return vols

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

