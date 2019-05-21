from textblob import TextBlob
import pandas as pd

df1 = pd.DataFrame()
df2 = pd.DataFrame()
with open('merged.txt') as fp:
    for line in fp:
        line = line.replace('They fight crime!','')
        if("She's" in line):
            line = line.split("She's")[1]
            df1 = df1.append({'Character':line,'Polarity':TextBlob(line).sentiment.polarity}, ignore_index=True)
        if("he's" in line):
            line = line.split("he's")[1]
            df2 = df2.append({'Character':line,'Polarity':TextBlob(line).sentiment.polarity}, ignore_index=True)
df1 = df1.sort_values(by=['Polarity'])
df2 = df2.sort_values(by=['Polarity'])
worst_string = "He's "+df2.head(1)['Character'].values[0].strip()+" She's "+df1.head(1)['Character'].values[0].strip()+" They fight crime!"
best_string = "He's "+df2.tail(1)['Character'].values[0].strip()+" She's "+df1.tail(1)['Character'].values[0].strip()+" They fight crime!"
print(worst_string)
print(best_string)
