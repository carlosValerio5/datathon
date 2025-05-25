{{ config(materialized='table') }}

select *
from read_csv_auto('data/tabla_maestra_completa.csv', HEADER=TRUE)