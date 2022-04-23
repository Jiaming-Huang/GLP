clear 
import delimited "F:\Research\GLP\Code\data\MSA_features_Adhoc_FE_Ylag.csv"
gen inc = log(pincpc)
gen popu = log(pop)
gen gdp = log(rgdppc)
gen empl = log(emp)


corr inc popu gdp empl d2i_low d2i_h wrluri elasticity fe_y_g3
mlogit fe_y_g3 gdp, baseoutcome(1)
eststo
mlogit fe_y_g3 empl, baseoutcome(1)
eststo
mlogit fe_y_g3 d2i_low, baseoutcome(1)
eststo
mlogit fe_y_g3 d2i_low elasticity, baseoutcome(1)
eststo
mlogit fe_y_g3 gdp d2i_low elasticity, baseoutcome(1)
eststo

esttab using tmp3.tex
drop _est_est1 _est_est2 _est_est3 _est_est4 _est_est5 _est_est6

mlogit fe_y_g4 gdp, baseoutcome(1)
eststo
mlogit fe_y_g4 empl, baseoutcome(1)
eststo
mlogit fe_y_g4 d2i_low, baseoutcome(1)
eststo
mlogit fe_y_g4 d2i_low elasticity, baseoutcome(1)
eststo
mlogit fe_y_g4 gdp d2i_low elasticity, baseoutcome(1)
eststo

esttab using tmp4.tex

