filename="salestransslice1.xlsx"
importrows=1000

# binning.  no columns or rows are excluded
ycolno=9
ybins=[0,8,16,20000]
Xbins=[[],
       [],
       [],
       [],
       [],
       [],
       [],
       [],
       [],
       [],
       [],
       [0,1,2,5,10,50,100,500,1000]]

# convert dates into day count changes?
day_deltas=True   
# date field to delta must be called 'date'

###########################
# cant exclude the 0 column
excludecols=[2,3,4,5,6,7,9,10,12,13,14]  # this is for the coding and decoding (after binning and day delta conversion)

