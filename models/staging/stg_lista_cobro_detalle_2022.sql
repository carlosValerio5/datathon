-- stg_lista_cobro_detalle_2022.sql
{{ config(materialized='table') }}
select * from read_csv_auto('data/ListaCobroDetalle2022.csv', HEADER=TRUE)
