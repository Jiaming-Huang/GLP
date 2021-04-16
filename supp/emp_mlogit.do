clear 
import excel "F:\Research\GLP\data\MSA_Feature.xlsx", sheet("FE_Y") firstrow case(lower) clear
gen inc = log(pincpc)
gen popu = log(pop)
gen gdp = log(rgdppc)
gen empl = log(emp)


corr inc popu gdp empl d2i_low d2i_h wrluri elasticity
mlogit g3 inc, baseoutcome(1)
eststo
mlogit g3 empl, baseoutcome(1)
eststo
mlogit g3 inc empl, baseoutcome(1)
eststo
mlogit g3 d2i_low d2i_h, baseoutcome(1)
eststo
mlogit g3 d2i_h wrluri elasticity, baseoutcome(1)
eststo
mlogit g3 inc d2i_h elasticity, baseoutcome(1)
eststo

esttab using tmp3.tex

mlogit g4 inc, baseoutcome(1)
eststo
mlogit g4 empl, baseoutcome(1)
eststo
mlogit g4 inc empl, baseoutcome(1)
eststo
mlogit g4 d2i_low d2i_h, baseoutcome(1)
eststo
mlogit g4 d2i_h wrluri elasticity, baseoutcome(1)
eststo
mlogit g4 inc d2i_h elasticity, baseoutcome(1)
eststo

esttab using tmp4.tex

