import pandas as pd
from datetime import datetime
TX_TI=pd.read_csv("TX_TI.csv")
for i in range(len(TX_TI)):
    print(TX_TI['Date'][i])
    x=datetime.strptime(TX_TI['Date'][i],'%Y/%m/%d')
    x=datetime.strftime(x,'%Y-%m-%d')
    
    TX_TI['Date'][i]=x
TX_TI=TX_TI.drop(['Unnamed: 0'],axis=1)
print(TX_TI)
TX_TI.to_csv("TX_TIWithnewdate.csv")