-- stg_lista_cobro_emisora.sql
{{ config(materialized='table') }}
select * from read_csv_auto('data/ListaCobroEmisora.csv', HEADER=TRUE)
