#Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

#import data
df = pd.read_csv("iris.data")
df.columns = ['Sepal Length','Sepal Width','Petal Length','Petal Width','class']

#Numero total de registros
print('Numero total de registros: ' + str(df.shape[0]))

#Drop duplicated data (cleaning data)
# df = df.drop_duplicates()
# print('Numero total de registros sin duplicados: ' + str(df.shape[0]))

#Plot data
ax = df['class'].value_counts().plot(kind='bar')
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()))

#Show data
plt.title("Flores por Tipo")
plt.ylabel("Numero de muestras")
plt.show()
