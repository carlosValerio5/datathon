-- Use the `ref` function to select from other models

select *
from "credifiel"."analytics"."my_first_dbt_model"
where id = 1