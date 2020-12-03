#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:02:59 2020

@author: tonedogga
"""

import numpy as np
import matplotlib.pyplot as plt  
 

def urgency(z,maxx,maxy,urgency_cross):
    return maxy-(maxy / (1+np.exp(-z+(urgency_cross))))



def efficiency(z,maxx,maxy,efficiency_cross):
    return (maxy / (1 + np.exp(-z+(efficiency_cross))))


# def seasonality(z,length):
#     xmas_start_day=250
#     base=np.full((total_days_in_year),0.9)
#     xmas_up=np.r_[np.full((xmas_start_day),0),np.tan((z[xmas_start_day:]-xmas_start_day)/15)]
#     #s=np.r_[base,xmas_up]
#     #print(s)
    
#     return np.add(base,xmas_up)

def area_under_curve(curve,startx,endx):
    return np.trapz(curve[startx:endx+1])


def seasonality(z,length):
    amp=6
    offset=250
    curve_pos=210
    wl=1.3
    xmas_up=1.1+np.sin(((-z[:-curve_pos]-0.15)/wl)+offset)/amp
    base=np.full((curve_pos),0.932)
    s=np.r_[base,xmas_up]
    return s


#def logit(z):
#    return 1 / (1 + np.exp(-z))
  
# x=np.linspace(0,3,1000).reshape(-1,1)
# print("x",x)
# y=logit(x)
# print("y",y)


urgency_maxx=10
urgency_maxy=2
urgency_cross=4


efficiency_maxx=50
efficiency_maxy=2.1
efficiency_cross=7

total_days_in_year=366
total_maxx=101
total_graph_xpoints=10
z = np.linspace(0, total_graph_xpoints, total_maxx)

#logits=np.c_[z,logit(z,maxx,maxy)]
urgency_curve=urgency(z,urgency_maxx,urgency_maxy,urgency_cross)
efficiency_curve=efficiency(z,efficiency_maxx,efficiency_maxy,efficiency_cross)

urgency_curve=np.r_[urgency_curve,np.full((total_days_in_year-total_maxx), 0)]
efficiency_curve=np.r_[efficiency_curve,np.full((total_days_in_year-total_maxx), 2)]

z2 = np.linspace(0, urgency_maxx, total_days_in_year)

seasonality_curve=seasonality(z2,total_days_in_year)   #np.ones((total_days_in_year))





print("area under curve urgency",area_under_curve(urgency_curve,0,366))
print("area under curve efficiency",area_under_curve(efficiency_curve,0,100))
print("area under curve seasonality",area_under_curve(seasonality_curve,0,366))

#print(urgency_curve,urgency_curve.shape)
#print(efficiency_curve,efficiency_curve.shape)

curves=np.c_[urgency_curve,efficiency_curve,seasonality_curve]

#print(curves,curves.shape)

# z = np.linspace(0, 10, 200)

# plt.plot([0, 10], [0, 0], 'k-')
# plt.plot([0, 10], [1, 1], 'k--')
# plt.plot([0, 0], [-0.2, 2.2], 'k-')
# #plt.plot([-5, 5], [-3/4, 7/4], 'g--')
#plt.plot(curves, "b-", linewidth=2)
plt.plot(curves[:,0], "b-", linewidth=2)
plt.plot(curves[:,1], "r-", linewidth=2)
plt.plot(curves[:,2], "g-", linewidth=2)

# #props = dict(facecolor='black', shrink=0.1)
# #plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1), arrowprops=props, fontsize=14, ha="center")
# #plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0), arrowprops=props, fontsize=14, ha="center")
# ##plt.annotate('Linear', xytext=(2, 0.2), xy=(0, 0.5), arrowprops=props, fontsize=14, ha="center")
plt.grid(True)
plt.title("Urgency (blue) vs efficiency (red) vs seasonality curve (green)", fontsize=11)
plt.xlabel("days to out of stock / % of ideal manu run / days of year",fontsize=10)

# plt.axis([0, 10, -0.2, 2.2])

# save_fig("sigmoid_saturation_plot")
plt.show()

