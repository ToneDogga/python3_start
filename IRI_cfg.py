# IRI_cfg.py
# constants for IRI_reader_vx-xx.py
#



# { spreadsheet number : spreadsheet name }
infilenamedict=dict({0:"IRI_WW_jams_v11.xlsx",
                     1:"IRI_coles_jams_v11.xlsx",
                     2:"IRI_coles_jams_v12.xlsx",
                     3:"IRI_coles_jams_v13.xlsx"
                    })

#  { queryno : [spreadsheet number,[list of column names]]}
querydict=dict({0: [0,["1","15","29"]],
                1: [1,["48","62","44","58","54","55","68","69"]],
                2: [1,["43","47","57","61"]],
                3: [0,["71","72","85","86","99","100"]],
                4: [2,["5","86","99","100"]],
                5: [3,["7","8","18"]]
                    })




logfile="IRI_reader_logfile.txt"
resultsfile="IRI_reader_results.txt"
pklsave="IRI_savenames.pkl"
colnamespklsave="IRI_savecoldetails.pkl"
fullcolnamespklsave="IRI_saveallcoldetails.pkl"

column_zero_name="0"   #scan week"
startdatestr='2018/07/14'
finishdatestr='2022/07/01'

