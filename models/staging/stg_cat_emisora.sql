{{ config(materialized='table') }}

select *
from read_csv_auto('data/CatEmisora.csv', HEADER=TRUE)