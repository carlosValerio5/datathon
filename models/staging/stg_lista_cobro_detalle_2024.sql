-- stg_lista_cobro_detalle_2024.sql
{{ config(materialized='table') }}
select * from read_csv_auto('data/ListaCobroDetalle2024.csv', HEADER=TRUE)
