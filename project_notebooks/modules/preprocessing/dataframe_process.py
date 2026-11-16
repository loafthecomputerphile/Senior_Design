import polars as pl

def filter_coughvid_df(df: pl.DataFrame) -> pl.DataFrame:
  return df.filter(
    (pl.col("cough_detected") > 0.8)
    & (pl.col("age").is_not_null())
    & (pl.col("gender").is_not_null())
    & (pl.col("status").is_not_null())
  )
  
def filter_coughvid_df_yamnet(df: pl.DataFrame) -> pl.DataFrame:
  return df.filter(
    (pl.col("age").is_not_null())
    & (pl.col("gender").is_not_null())
    & (pl.col("status").is_not_null())
  )

# primary-expert diagnosis -> the 3 training classes. Infections and obstructive
# disease are symptomatic coughs; only expert-labeled rows are kept at all.
EXPERT_DIAGNOSIS_MAP: dict[str, str] = {
  "healthy_cough": "healthy",
  "COVID-19": "COVID-19",
  "lower_infection": "symptomatic",
  "upper_infection": "symptomatic",
  "obstructive_disease": "symptomatic",
}

def filter_coughvid_df_expert(df: pl.DataFrame) -> pl.DataFrame:
  """Expert-annotated subset only: status is derived from the primary expert's
  diagnosis (metadata_compiled.csv), replacing self-reported status entirely."""
  return (
    df.filter(pl.col("diagnosis_1").is_in(list(EXPERT_DIAGNOSIS_MAP.keys())))
      .with_columns(
        pl.col("diagnosis_1").replace_strict(EXPERT_DIAGNOSIS_MAP).alias("status"),
        pl.col("cough_detected").cast(pl.Float64),
      ).filter(
        (pl.col("age").is_not_null())
        & (pl.col("gender").is_not_null())
        & (pl.col("status").is_not_null())
      )
  )