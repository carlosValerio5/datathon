-- stg_lista_cobro.sql
{{ config(materialized='table') }}
select * from read_csv_auto('data/ListaCobro.csv', HEADER=TRUE)
