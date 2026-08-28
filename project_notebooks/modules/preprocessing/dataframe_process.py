import polars as pl

def filter_coughvid_df(df: pl.DataFrame) -> pl.DataFrame:
  return df.filter(
    (pl.col("cough_detected") > 0.8)
    & (pl.col("age").is_not_null())
    & (pl.col("gender").is_not_null())
    & (pl.col("status").is_not_null())
  )