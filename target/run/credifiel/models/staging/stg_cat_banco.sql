
  
    
    

    create  table
      "credifiel"."analytics"."stg_cat_banco__dbt_tmp"
  
    as (
      

select *
from read_csv_auto('data/CatBanco.csv', HEADER=TRUE)
    );
  
  