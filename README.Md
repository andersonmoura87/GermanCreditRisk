Source:

Ulrike GrÃ¶mping
Beuth University of Applied Sciences Berlin
Website with contact information: https://prof.beuth-hochschule.de/groemping/


Data Set Information:

The widely used Statlog German credit data as of November 2019, suffers from severe errors in the coding information and does not come with any background information. The 'South German Credit' data provide a correction and some background information, based on the Open Data LMU (2010) representation of the same data and several other German language resources.


Attribute Information:

This section contains a brief description for each attribute. Details on attribute coding can be obtained from the accompanying R code for reading the data or the accompanying code table, as well as from Groemping (2019) (listed under 'Relevant Papers').


Variable name: account_status
Content: status of the debtor's checking account with the bank (categorical)

     1 :  no checking account                       
     2 : ... < 0 DM                                
     3 : 0<= ... < 200 DM                          
     4 : ... >= 200 DM / salary for at least 1 year


Variable name: credit_duration
Content: credit duration in months (quantitative)



Variable name: history_of_compliance
Content: history of compliance with previous or concurrent credit contracts (categorical)

     0 : delay in paying off in the past            
     1 : critical account/other credits elsewhere   
     2 : no credits taken/all credits paid back duly
     3 : existing credits paid back duly till now   
     4 : all credits at this bank paid back duly    


Variable name: credit_purpose 
Content: purpose for which the credit is needed (categorical)

     0 : others             
     1 : car (new)          
     2 : car (used)         
     3 : furniture/equipment
     4 : radio/television   
     5 : domestic appliances
     6 : repairs            
     7 : education          
     8 : vacation           
     9 : retraining         
     10 : business 
 
 
Variable name: credit_amount 
Content: credit amount in DM (quantitative; result of monotonic transformation; actual data and type of transformation unknown)


Variable name: savings
Content: debtor's savings (categorical)

     1 : unknown/no savings account
     2 : ... <  100 DM             
     3 : 100 <= ... <  500 DM      
     4 : 500 <= ... < 1000 DM      
     5 : ... >= 1000 DM    


Variable name: employment_duration
Content: duration of debtor's employment with current employer (ordinal; discretized quantitative)

     1 : unemployed      
     2 : < 1 yr          
     3 : 1 <= ... < 4 yrs
     4 : 4 <= ... < 7 yrs
     5 : >= 7 yrs        


Variable name: installment_rate
Content: credit installments as a percentage of debtor's disposable income (ordinal; discretized quantitative)

     1 : >= 35         
     2 : 25 <= ... < 35
     3 : 20 <= ... < 25
     4 : < 20


Variable name: personal_status_sex
Content: combined information on sex and marital status; categorical; sex cannot be recovered from the
variable, because male singles and female non-singles are coded with the same code (2); female widows cannot
be easily classified, because the code table does not list them in any of the female categories

     1 : male : divorced/separated           
     2 : female : non-single or male : single
     3 : male : married/widowed              
     4 : female : single  


Variable name: other_debtors
Content: Is there another debtor or a guarantor for the credit? (categorical)

     1 : none        
     2 : co-applicant
     3 : guarantor   


Variable name: present_residence
Content: length of time (in years) the debtor lives in the present residence (ordinal; discretized quantitative)


     1 : < 1 yr          
     2 : 1 <= ... < 4 yrs
     3 : 4 <= ... < 7 yrs
     4 : >= 7 yrs    


Variable name: property
Content: the debtor's most valuable property, i.e. the highest possible code is used. Code 2 is used, if codes 3
or 4 are not applicable and there is a car or any other relevant property that does not fall under variable
sparkont. (ordinal)

     1 : unknown / no property                    
     2 : car or other                             
     3 : building soc. savings agr./life insurance
     4 : real estate   
 

Variable name: age
Content: age in years (quantitative)

 


Variable name: other_installment_plans
Content: installment plans from providers other than the credit-giving bank (categorical)

     1 : bank  
     2 : stores
     3 : none 


Variable name: type_of_housing
Content: type of housing the debtor lives in (categorical)

     1 : for free
     2 : rent    
     3 : own     


Variable name: number_credits
Content: number of credits including the current one the debtor has (or had) at this bank (ordinal, discretized
quantitative); contrary to Fahrmeir and HamerleÃ¢â‚¬â„¢s (1984) statement, the original data values are not available.

     1 : 1   
     2 : 2-3 
     3 : 4-5 
     4 : >= 6



Variable name: job
Content: quality of debtor's job (ordinal)

     1 : unemployed/unskilled - non-resident       
     2 : unskilled - resident                      
     3 : skilled employee/official                 
     4 : manager/self-empl./highly qualif. employee


Variable name: people_liable
Content: number of persons who financially depend on the debtor (i.e., are entitled to maintenance) (binary,
discretized quantitative)

     1 : 3 or more
     2 : 0 to 2   


Variable name: telephone
Content: Is there a telephone landline registered on the debtor's name? (number)


Variable name: level_of_education
Content: level of education (categorical)


Variable name: entry_payment
Content: entry payment in DM (quantitative; result of monotonic transformation; actual data and type of transformation unknown)



Variable name: credit_risk
Content: Has the credit contract been complied with (good) or not (bad) ? (binary)


     0 : bad 
     1 : good



Relevant Papers:

Fahrmeir, L. and Hamerle, A. (1981, in German). Kategoriale Regression in der betrieblichen Planung. *Zeitschrift für Operations Research* **25**, B63-B78.

Fahrmeir, L. and Hamerle, A. (1984, in German). *Multivariate Statistische Verfahren* (1st ed., Ch.8 and Appendix C). De Gruyter, Berlin.

Grömping, U. (2019). South German Credit Data: Correcting a Widely Used Data Set. Report 4/2019, Reports in Mathematics, Physics and Chemistry, Department II, Beuth University of Applied Sciences Berlin.

Häußler, W.M. (1979, in German). Empirische Ergebnisse zu Diskriminationsverfahren bei Kreditscoringsystemen. *Zeitschrift für Operations Research* **23**, B191-B210.

Hofmann, H.J. (1990, in German). Die Anwendung des CART-Verfahrens zur statistischen Bonitätsanalyse von Konsumentenkrediten. *Zeitschrift für Betriebswirtschaft* **60**, 941-962.

Open data LMU (2010; accessed Nov 27 2019; in German). Kreditscoring zur Klassifikation von Kreditnehmern.



Citation Request:

Grömping, U. (2019). South German Credit Data: Correcting a Widely Used Data Set. Report 4/2019, Reports in Mathematics, Physics and Chemistry, Department II, Beuth University of Applied Sciences Berlin.