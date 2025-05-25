
  
    
    

    create  table
      "credifiel"."analytics"."mart_maximizacion_cobranza__dbt_tmp"
  
    as (
      -- Modelo analítico para maximización de cobranza
-- mart_maximizacion_cobranza.sql



WITH cobros_enriquecidos AS (
    SELECT * FROM "credifiel"."analytics"."int_cobros_enriquecidos"
),

-- Obtener información de emisoras y bancos
emisoras AS (
    SELECT * FROM "credifiel"."analytics"."stg_cat_emisora"
),

tabla_maestra AS (
    SELECT * FROM "credifiel"."analytics"."stg_tabla_maestra_completa"
),

-- Análisis de secuencias de cobro
secuencias_cobro AS (
    SELECT
        idCredito,
        idBanco,
        idRespuestaBanco,
        fechaEnvioCobro,
        ROW_NUMBER() OVER(PARTITION BY idCredito ORDER BY fechaEnvioCobro) AS intento_numero,
        LEAD(idRespuestaBanco) OVER(PARTITION BY idCredito ORDER BY fechaEnvioCobro) AS siguiente_respuesta,
        cobro_exitoso
    FROM cobros_enriquecidos
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
    -- Métricas de efectividad para maximización
    COUNT(*) AS total_intentos,
    SUM(c.cobro_exitoso) AS total_exitosos,
    SUM(c.cobro_exitoso)::FLOAT / NULLIF(COUNT(*), 0) AS tasa_exito,
    SUM(c.montoCobrado) AS monto_recuperado,
    SUM(c.montoCobrar) AS monto_intentado,
    SUM(c.montoCobrado)::FLOAT / NULLIF(SUM(c.montoCobrar), 0) AS tasa_recuperacion_monto,
    -- Análisis de secuencias exitosas
    SUM(CASE WHEN sc.intento_numero = 1 AND sc.cobro_exitoso = 1 THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN sc.intento_numero = 1 THEN 1 ELSE 0 END), 0) AS tasa_exito_primer_intento,
    SUM(CASE WHEN sc.intento_numero > 1 AND sc.cobro_exitoso = 1 THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN sc.intento_numero > 1 THEN 1 ELSE 0 END), 0) AS tasa_exito_intentos_posteriores,
    -- Probabilidad de éxito después de cada tipo de respuesta
    SUM(CASE WHEN sc.idRespuestaBanco = '01' AND sc.siguiente_respuesta = '00' THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN sc.idRespuestaBanco = '01' THEN 1 ELSE 0 END), 0) AS prob_exito_tras_cuenta_inexistente,
    SUM(CASE WHEN sc.idRespuestaBanco = '02' AND sc.siguiente_respuesta = '00' THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN sc.idRespuestaBanco = '02' THEN 1 ELSE 0 END), 0) AS prob_exito_tras_cuenta_bloqueada,
    SUM(CASE WHEN sc.idRespuestaBanco = '03' AND sc.siguiente_respuesta = '00' THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN sc.idRespuestaBanco = '03' THEN 1 ELSE 0 END), 0) AS prob_exito_tras_cuenta_cancelada,
    -- Índice de potencial de recuperación (para maximización)
    (SUM(c.cobro_exitoso)::FLOAT / NULLIF(COUNT(*), 0)) *
    (SUM(c.montoCobrado)::FLOAT / NULLIF(SUM(c.montoCobrar), 0)) *
    (CASE WHEN tm.tipo_cobro = 'TOTAL Y PARCIAL' THEN 1.2 ELSE 1.0 END) AS indice_potencial_recuperacion
FROM cobros_enriquecidos c
LEFT JOIN emisoras e ON c.idBanco = e.IdBanco
LEFT JOIN tabla_maestra tm ON c.idBanco = tm.idBanco AND e.idEmisora = tm.idEmisora
LEFT JOIN secuencias_cobro sc ON c.idCredito = sc.idCredito AND c.idBanco = sc.idBanco AND c.fechaEnvioCobro = sc.fechaEnvioCobro
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
ORDER BY indice_potencial_recuperacion DESC
    );
  
  