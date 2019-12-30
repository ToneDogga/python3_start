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
infilenamedict=dict({0:["IRI_WW_jams_v11.xlsx","p_ww_bl"],
                     1:["IRI_coles_jams_v11.xlsx","p_coles_bl"],
                     2:["IRI_WW_jams_v11.xlsx","p_ww_total"],
                     3:["IRI_coles_jams_v11.xlsx","p_coles_total"],
                     4:["IRI_WW_jams_v11.xlsx","ww_sd_bl"],
                     5:["IRI_coles_jams_v11.xlsx","coles_sd_bl"],
                     6:["IRI_WW_jams_v11.xlsx","ww_sd_total"],
                     7:["IRI_coles_jams_v11.xlsx","coles_sd_total"],
                     8:["IRI_WW_jams_v11.xlsx","ww_sd_dod"],
                     9:["IRI_coles_jams_v11.xlsx","coles_sd_dod"],
                     10:["IRI_WW_jams_v11.xlsx","ww_dod"],
                     11:["IRI_coles_jams_v11.xlsx","coles_dod"],
                     12:["IRI_WW_jams_v11.xlsx","ww_main_bl"],
                     13:["IRI_coles_jams_v11.xlsx","coles_main_bl"],
                     14:["IRI_WW_jams_v11.xlsx","ww_bb_dod"],
                     15:["IRI_coles_jams_v11.xlsx","coles_bb_dod"],
                     16:["IRI_WW_jams_v11.xlsx","ww_total"],
                     17:["IRI_coles_jams_v11.xlsx","coles_total"],
                     18:["IRI_WW_jams_v11.xlsx","ww_premium_dod"],
                     19:["IRI_coles_jams_v11.xlsx","coles_premium_dod"],
                     20:["IRI_WW_jams_v11.xlsx","ww_main_dod"],
                     21:["IRI_coles_jams_v11.xlsx","coles_main_dod"]

                    })

# plot number, list of queryies to plot, corr and scatter
plotdict=dict({0: ["ww_dod","coles_dod","ww_sd_dod","coles_sd_dod"],
               1: ["p_coles_total","p_ww_total","ww_total","coles_total"],
               2: ["ww_premium_dod","coles_premium_dod","ww_main_dod","coles_main_dod","ww_dod","coles_dod"]
               })



#  { queryno : [spreadsheet number,[list of column names]]}
querydict=dict({0: [0,["36"]], # ,"15","29"]],
                1: [1,["36"]],
                2: [2,["29"]],
                3: [3,["29"]],
                4: [4,["57"]],
                5: [5,["57"]],
                6: [6,["64"]],
                7: [7,["64"]],
                8: [8,["61"]],
                9: [9,["61"]],
                10: [10,["5"]],
                11: [11,["5"]],
                12: [12,["22"]],
                13: [13,["22"]],
                14: [14,["47"]],
                15: [15,["47"]],
                16: [16,["1"]],
                17: [17,["1"]],
                18: [18,["33"]],
                19: [19,["33"]],
                20: [20,["19"]],
                21: [21,["19"]]
                
                })




logfile="IRI_reader_logfile.txt"
resultsfile="IRI_reader_results.txt"
pklsave="IRI_savenames.pkl"
colnamespklsave="IRI_savecoldetails.pkl"
fullcolnamespklsave="IRI_saveallcoldetails.pkl"
dfdictpklsave="IRI_savedfdict.pkl"

column_zero_name="0"   #scan week"
startdatestr='2018/07/14'
finishdatestr='2022/07/01'

