Part 1: main data for empirical application

Input:
MSAs_SA.xls: house price index from Freddie Mac
data_HFI.xls: high frequency MP surprises shared by Geert Mesters
Instruments_web.xlsx: information-robust IVs (Miranda-Agrippino & Ricco, 2016)
Monetarydat.xlsx: RR shocks from Ramey's website
FRED-MD.csv: monthly macro series from FRED-MD database
historicalweeklydata.xls: FRM 30 from Freddie Mac

Intermediate output:
HPI.csv: house price index
macro.csv: selected macro series
FRM30.csv: weekly FRM30

Output:
empirical_main.csv

From input to intermediate output it's simple; from intermediate to final output I use python to merge different data sources.

=====================================
Part 2: MSA features

Input:
Group_EST.csv: estimated group membership by the GLP
EconProfile_MSA.csv: CAINC30 Economic Profile from BEA
GDP_MSA.csv: Real GDP, from BEA
household-debt-by-msa.csv: household debt-to-income ratio, from FED
SUPPLYDATA.zip: Wharton regulation index & supply elasiticity from Saiz (http://real.wharton.upenn.edu/~saiz/)

Intermediate output:
EconProfile_MSA.csv
GDP_MSA.csv
household-debt-by-msa.csv
SUPPLYDATA.dta

Output:
MSA_group_feature.csv

=====================================
Part 3: Plotting

Input:
list1_Sep_2018.xls: MSAs to county crosswalk by OMB
geojson-counties-fips.json

Intermediate output:
cbsa_county.csv

merge cbsa_county.csv and MSA_group_feature.csv