-- stg_cat_respuesta_bancos.sql

select * from read_csv_auto('data/CatRespuestaBancos.csv', HEADER=TRUE)