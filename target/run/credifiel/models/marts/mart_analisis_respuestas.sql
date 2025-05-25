
  
    
    

    create  table
      "credifiel"."analytics"."mart_analisis_respuestas__dbt_tmp"
  
    as (
      -- Modelo analítico para análisis de respuestas de bancos
-- mart_analisis_respuestas.sql



WITH cobros_enriquecidos AS (
    SELECT * FROM "credifiel"."analytics"."int_cobros_enriquecidos"
)

SELECT
    idRespuestaBanco,
    descripcion_respuesta,
    COUNT(*) AS total_ocurrencias,
    SUM(montoCobrar) AS monto_total_intentado,
    SUM(montoCobrado) AS monto_total_recuperado,
    SUM(montoCobrado)::FLOAT / NULLIF(SUM(montoCobrar), 0) AS tasa_recuperacion_monto,
    -- Distribución por banco
    COUNT(DISTINCT idBanco) AS cantidad_bancos,
    COUNT(DISTINCT idCredito) AS cantidad_clientes,
    -- Tendencia temporal
    SUM(CASE WHEN anio_origen = '2022' THEN 1 ELSE 0 END) AS ocurrencias_2022,
    SUM(CASE WHEN anio_origen = '2023' THEN 1 ELSE 0 END) AS ocurrencias_2023,
    SUM(CASE WHEN anio_origen = '2024' THEN 1 ELSE 0 END) AS ocurrencias_2024,
    SUM(CASE WHEN anio_origen = '2025' THEN 1 ELSE 0 END) AS ocurrencias_2025,
    -- Proporción de ocurrencias por año
    SUM(CASE WHEN anio_origen = '2022' THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(*), 0) AS proporcion_2022,
    SUM(CASE WHEN anio_origen = '2023' THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(*), 0) AS proporcion_2023,
    SUM(CASE WHEN anio_origen = '2024' THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(*), 0) AS proporcion_2024,
    SUM(CASE WHEN anio_origen = '2025' THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(*), 0) AS proporcion_2025
FROM cobros_enriquecidos
GROUP BY idRespuestaBanco, descripcion_respuesta
ORDER BY total_ocurrencias DESC
    );
  
  