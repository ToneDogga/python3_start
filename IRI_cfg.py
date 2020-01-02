# IRI_cfg.py
# constants for IRI_reader_vx-xx.py
#
# promotional activity and brand interaction matrix
##Good
##Expanding market share
##Steal from competitors
##
##Ok
##Channel switch (Coles-> ww)
##Trade up or down
##
##Bad
##Brand position and price point erosion
##Pantry loading
#
# Hypothesis
# 1) St Dalfours WW sales are gaining. incremental sales are huge during promotions.
# where is the growth coming from?
# possibilities to test:
# a) total category gain
# b) WW category gain
# c) coles category loss

#  promotional intensity

# d) total premium category gain
# e) WW premium category gain
# f) coles premium category loss
# g) total premium category depth of distribution gain
# h) WW premium category depth of distribution gain
# i) coles premium category depth of distribution loss
# j) total varietal growth ie Strawberry jam gain
# k) WW varietal growth ie strawberry jam gain
# l) coles varietal loss
# m) total Premium competitors baseline gain
# n) WW premium competitors baseline gain
# o) coles premium competitors baseline loss
# p) total Premium competitiors incremental gain
# q) WW premium compeitors incremental gain
# r) coles premium competitors incremental loss
# s) total Premium competitiors total gain
# t) WW premium compeitors total gain
# u) coles premium competitors total loss
# v) total Mainstream competitors baseline gain
# w) total Mainstream competitors incremental gain
# x) total Mainstream competitors total gain
# y) WW Mainstream competitors baseline gain
# z) WW Mainstream competitors incremental gain
# aa) WW Mainstream competitors total gain
# ab) coles Mainstream competitors baseline loss
# ac) coles Mainstream competitors incremental loss
# ad) coles Mainstream competitors total loss
#
#


# { spreadsheet number : [spreadsheet name, alias_for_column_names] }
infilenamedict=dict({0:["IRI_WW_jams_v11.xlsx","ww_units_total"],
                     1:["IRI_coles_jams_v11.xlsx","coles_units_total"],
                     2:["IRI_WW_jams_v11.xlsx","ww_units_mainstream"],
                     3:["IRI_coles_jams_v11.xlsx","coles_units_mainstream"],
                     4:["IRI_WW_jams_v11.xlsx","ww_units_premium"],
                     5:["IRI_coles_jams_v11.xlsx","coles_units_premium"],
                     6:["IRI_WW_jams_v11.xlsx","ww_total_dod"],
                     7:["IRI_coles_jams_v11.xlsx","coles_total_dod"],
                     8:["IRI_WW_jams_v11.xlsx","ww_mainstream_dod"],
                     9:["IRI_coles_jams_v11.xlsx","coles_mainstream_dod"],
                     10:["IRI_WW_jams_v11.xlsx","ww_premium_dod"],
                     11:["IRI_coles_jams_v11.xlsx","coles_premium_dod"],
                     12:["IRI_WW_jams_v11.xlsx","ww_units_baseline_mainstream"],
                     13:["IRI_coles_jams_v11.xlsx","coles_units_baseline_mainstream"],
                     14:["IRI_WW_jams_v11.xlsx","ww_units_baseline_premium"],
                     15:["IRI_coles_jams_v11.xlsx","coles_units_baseline_premium"],
                     16:["IRI_WW_jams_v11.xlsx","ww_units_incremental_mainstream"],
                     17:["IRI_coles_jams_v11.xlsx","coles_units_incremental_mainstream"],
                     18:["IRI_WW_jams_v11.xlsx","ww_units_incremental_premium"],
                     19:["IRI_coles_jams_v11.xlsx","coles_units_incremental_premium"],
                     20:["IRI_WW_jams_v11.xlsx","ww_percentunits_mainstream"],
                     21:["IRI_coles_jams_v11.xlsx","coles_percentunits_mainstream"],
                     22:["IRI_WW_jams_v11.xlsx","ww_percentunits_premium"],
                     23:["IRI_coles_jams_v11.xlsx","coles_percentunits_premium"]

                    })

# plot number, list of queryies to plot, corr and scatter
plotdict=dict({0: ["ww_units_total","coles_units_total","smooth_ww","smooth_coles"],
               1: ["ww_total_dod","coles_total_dod"],
               2: ["ww_units_mainstream","ww_units_premium","smooth_ww_mainstream","smooth_ww_premium","ww_total_mshare"],
               3: ["ww_mainstream_dod","ww_premium_dod"],
               4: ["coles_units_mainstream","coles_units_premium","smooth_coles_mainstream","smooth_coles_premium","coles_total_mshare"],
               5: ["coles_mainstream_dod","coles_premium_dod"],
               6: ["ww_units_mainstream","coles_units_mainstream","ww_mainstream_mshare","coles_mainstream_mshare"], 
               7: ["ww_units_premium","coles_units_premium","ww_premium_mshare","coles_premium_mshare"], 
               8: ["ww_units_baseline_mainstream","coles_units_baseline_mainstream"], 
               9: ["ww_units_baseline_premium","coles_units_baseline_premium"], 
               10: ["ww_units_incremental_mainstream","ww_percentunits_mainstream"], 
               11: ["coles_units_incremental_mainstream","coles_percentunits_mainstream"], 
               12: ["ww_units_incremental_premium","ww_percentunits_premium"], 
               13: ["coles_units_incremental_premium","coles_percentunits_premium"] 

               })



#  { queryno : [spreadsheet number,[list of column names]]}
querydict=dict({0: [0,["1"]], # ,"15","29"]],
                1: [1,["1"]],
                2: [2,["15"]],
                3: [3,["15"]],
                4: [4,["29"]],
                5: [5,["29"]],
                6: [6,["5"]],
                7: [7,["5"]],
                8: [8,["19"]],
                9: [9,["19"]],
                10: [10,["33"]],
                11: [11,["33"]],
                12: [12,["22"]],
                13: [13,["22"]],
                14: [14,["36"]],
                15: [15,["36"]],
                16: [16,["23"]],
                17: [17,["23"]],
                18: [18,["37"]],
                19: [19,["37"]],
                20: [20,["27"]],
                21: [21,["27"]],
                22: [22,["41"]],
                23: [23,["41"]]
                 
                })


usable_measures=[0,4,5,8,9,10,11,12,13]


logfile="IRI_reader_logfile.txt"
resultsfile="IRI_reader_results.txt"
pklsave="IRI_savenames.pkl"
colnamespklsave="IRI_savecoldetails.pkl"
fullcolnamespklsave="IRI_saveallcoldetails.pkl"
dfdictpklsave="IRI_savedfdict.pkl"
dfpklsave="IRI_fullspreadsheetsave.pkl"


column_zero_name="0"   #scan week"
startdatestr='2018/07/14'
finishdatestr='2022/07/01'

