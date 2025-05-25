-- stg_lista_cobro_detalle_2025.sql
{{ config(materialized='table') }}
select * from read_csv_auto('data/ListaCobroDetalle2025.csv', HEADER=TRUE)
