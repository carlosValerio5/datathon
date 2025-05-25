
  
    
    

    create  table
      "credifiel"."analytics"."stg_cat_emisora__dbt_tmp"
  
    as (
      SELECT *
FROM read_csv_auto('data/ExtraccionDomFinal/CatEmisora.csv')
    );
  
  