import  pandas as pd


def getDuplicateColumns(df):
    duplicate_column_names = set()
    for x in range(df.shape[1]):
        col = df.iloc[:, x]
        for y in range(x + 1, df.shape[1]):
            other_col = df.iloc[:, y]
            if col.equals(other_col):
                duplicate_column_names.add(df.columns.values[y])
    return list(duplicate_column_names)

