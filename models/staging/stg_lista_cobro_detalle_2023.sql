-- stg_lista_cobro_detalle_2023.sql
{{ config(materialized='table') }}
select * from read_csv_auto('data/ListaCobroDetalle2023.csv', HEADER=TRUE)
