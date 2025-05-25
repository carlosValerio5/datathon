
  
    
    

    create  table
      "credifiel"."analytics"."stg_cat_banco__dbt_tmp"
  
    as (
      SELECT *
FROM read_csv_auto('data/ExtraccionDomFinal/CatBanco.csv')
    );
  
  