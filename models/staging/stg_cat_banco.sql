{{ config(materialized='table') }}

select *
from read_csv_auto('data/CatBanco.csv', HEADER=TRUE)
