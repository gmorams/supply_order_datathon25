import pandas as pd
data = pd.read_csv("train.csv", sep=';', encoding='utf-8')

# Dividim en train i test
ids = data["ID"].unique()
print(len(ids), "IDs únics")
from sklearn.model_selection import train_test_split

train_ids, val_ids = train_test_split(
    ids,
    test_size=0.20,
    random_state=42
)

train_df = data[data["ID"].isin(train_ids)].reset_index(drop=True)
val_df   = data[data["ID"].isin(val_ids)].reset_index(drop=True)

def prep_test(val_df):
  val_demand = (
      val_df.groupby("ID")["weekly_demand"]
      .sum()
      .reset_index()
      .rename(columns={"weekly_demand": "demand"})
  )
  val_static = val_df.drop_duplicates("ID").drop(columns=["weekly_demand"])
  val_final = val_static.merge(val_demand, on="ID", how="inner")
  validation = val_final.drop(columns=["num_week_iso","weekly_sales","Production", "year"])
  return validation

test = prep_test(val_df)


