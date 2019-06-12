import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline

a1=pd.DataFrame(a,index=[0,1,2,3,4,5,6,7,8])
print(a1)                                                                                                                                                         b=a1.groupby(['Year'])['Salary'].sum()
print("total salary spent in year",b)
c=a1.groupby(['name'])['Salary'].sum()
print("the salary recieved by each emp",c)
d=a1.groupby(['Year','Department'])['Salary'].sum()
print("salary spent on each dept",d )
