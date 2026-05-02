import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    df = pd.read_csv('/home/voxdei/Downloads/iris.csv', delimiter = ',')
    return df

def clear_data(frame):
    df = frame.copy()
    cat_columns = ['species']
    num_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # Анализ и очистка данных

    df['species'].replace({'setosa': 0, 'versicolor': 1, 'virginica': 2}, inplace=True)
    
    df = df.reset_index(drop=True)  # обновим индексы в датафрейме DF. если бы мы прописали drop = False, то была бы еще одна колонка - старые индексы
    # Разделение данных на признаки и целевую переменную
    
    
    # Предварительная обработка категориальных данных
    # Порядковое кодирование. Обучение, трансформация и упаковка в df
    
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    df.to_csv("df_clear.csv")
    return True

clear_data(download_data())