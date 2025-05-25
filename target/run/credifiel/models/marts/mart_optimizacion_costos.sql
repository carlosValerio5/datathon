
  
    
    

    create  table
      "credifiel"."analytics"."mart_optimizacion_costos__dbt_tmp"
  
    as (
      -- Modelo analítico para optimización de costos de comisiones
-- mart_optimizacion_costos.sql



WITH cobros_enriquecidos AS (
    SELECT * FROM "credifiel"."analytics"."int_cobros_enriquecidos"
),

-- Obtener información de emisoras y bancos
emisoras AS (
    SELECT * FROM "credifiel"."analytics"."stg_cat_emisora"
),

tabla_maestra AS (
    SELECT * FROM "credifiel"."analytics"."stg_tabla_maestra_completa"
)

SELECT
    c.idBanco,
    c.nombre_banco,
    e.idEmisora,
    e.Nombre AS nombre_emisora,
    tm.tipo_servicio,
    tm.canal_envio,
    tm.tipo_envio,
    tm.tipo_cobro,
    tm.horario_carga,
    tm.rango_cobro_banco,
    -- Métricas de efectividad
    COUNT(*) AS total_intentos,
    SUM(c.cobro_exitoso) AS total_exitosos,
    SUM(c.cobro_exitoso)::FLOAT / NULLIF(COUNT(*), 0) AS tasa_exito,
    SUM(c.montoCobrado) AS monto_recuperado,
    SUM(c.montoCobrar) AS monto_intentado,
    SUM(c.montoCobrado)::FLOAT / NULLIF(SUM(c.montoCobrar), 0) AS tasa_recuperacion_monto,
    -- Métricas de costo-beneficio (simuladas)
    -- Nota: Aquí asumimos un costo por intento basado en el tipo de servicio
    -- En un entorno real, estos costos vendrían de una tabla de comisiones
    CASE
        WHEN tm.tipo_servicio = 'TRADICIONAL' THEN COUNT(*) * 2.5
        WHEN tm.tipo_servicio = 'INTERBANCA' THEN COUNT(*) * 3.0
        WHEN tm.tipo_servicio = 'PARCIAL' THEN COUNT(*) * 2.0
        WHEN tm.tipo_servicio = 'MATUTINO' THEN COUNT(*) * 2.2
        ELSE COUNT(*) * 2.5
    END AS costo_total_comisiones,
    -- Relación costo-beneficio
    CASE
        WHEN tm.tipo_servicio = 'TRADICIONAL' THEN COUNT(*) * 2.5
        WHEN tm.tipo_servicio = 'INTERBANCA' THEN COUNT(*) * 3.0
        WHEN tm.tipo_servicio = 'PARCIAL' THEN COUNT(*) * 2.0
        WHEN tm.tipo_servicio = 'MATUTINO' THEN COUNT(*) * 2.2
        ELSE COUNT(*) * 2.5
    END / NULLIF(SUM(c.montoCobrado), 0) AS costo_por_peso_recuperado,
    -- Índice de eficiencia (mayor es mejor)
    (SUM(c.montoCobrado) / NULLIF(
        CASE
            WHEN tm.tipo_servicio = 'TRADICIONAL' THEN COUNT(*) * 2.5
            WHEN tm.tipo_servicio = 'INTERBANCA' THEN COUNT(*) * 3.0
            WHEN tm.tipo_servicio = 'PARCIAL' THEN COUNT(*) * 2.0
            WHEN tm.tipo_servicio = 'MATUTINO' THEN COUNT(*) * 2.2
            ELSE COUNT(*) * 2.5
        END, 0
    )) AS indice_eficiencia
FROM cobros_enriquecidos c
LEFT JOIN emisoras e ON c.idBanco = e.IdBanco
LEFT JOIN tabla_maestra tm ON c.idBanco = tm.idBanco AND e.idEmisora = tm.idEmisora
GROUP BY
    c.idBanco,
    c.nombre_banco,
    e.idEmisora,
    e.Nombre,
    tm.tipo_servicio,
    tm.canal_envio,
    tm.tipo_envio,
    tm.tipo_cobro,
    tm.horario_carga,
    tm.rango_cobro_banco
ORDER BY indice_eficiencia DESC
    );
  
  