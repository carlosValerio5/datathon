-- Modelo analítico para análisis de comportamiento de clientes
-- mart_comportamiento_clientes.sql



WITH cobros_enriquecidos AS (
    SELECT * FROM "credifiel"."analytics"."int_cobros_enriquecidos"
)

SELECT
    idCredito,
    COUNT(*) AS total_intentos_cobro,
    SUM(cobro_exitoso) AS total_cobros_exitosos,
    SUM(cobro_exitoso)::FLOAT / COUNT(*) AS tasa_exito_cobros,
    SUM(montoCobrado) AS monto_total_recuperado,
    SUM(montoCobrar) AS monto_total_intentado,
    SUM(montoCobrado)::FLOAT / NULLIF(SUM(montoCobrar), 0) AS tasa_recuperacion_monto,
    MIN(CASE WHEN cobro_exitoso = 1 THEN fechaCobroBanco ELSE NULL END) AS fecha_primer_cobro_exitoso,
    MAX(CASE WHEN cobro_exitoso = 1 THEN fechaCobroBanco ELSE NULL END) AS fecha_ultimo_cobro_exitoso,
    COUNT(DISTINCT idBanco) AS cantidad_bancos_utilizados,
    -- Patrones de respuesta
    COUNT(DISTINCT idRespuestaBanco) AS cantidad_tipos_respuesta,
    SUM(CASE WHEN idRespuestaBanco = '01' THEN 1 ELSE 0 END) AS cuenta_inexistente_count,
    SUM(CASE WHEN idRespuestaBanco = '02' THEN 1 ELSE 0 END) AS cuenta_bloqueada_count,
    SUM(CASE WHEN idRespuestaBanco = '03' THEN 1 ELSE 0 END) AS cuenta_cancelada_count,
    -- Tendencia temporal
    SUM(CASE WHEN anio_origen = '2022' THEN cobro_exitoso ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN anio_origen = '2022' THEN 1 ELSE 0 END), 0) AS tasa_exito_2022,
    SUM(CASE WHEN anio_origen = '2023' THEN cobro_exitoso ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN anio_origen = '2023' THEN 1 ELSE 0 END), 0) AS tasa_exito_2023,
    SUM(CASE WHEN anio_origen = '2024' THEN cobro_exitoso ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN anio_origen = '2024' THEN 1 ELSE 0 END), 0) AS tasa_exito_2024,
    SUM(CASE WHEN anio_origen = '2025' THEN cobro_exitoso ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN anio_origen = '2025' THEN 1 ELSE 0 END), 0) AS tasa_exito_2025
FROM cobros_enriquecidos
GROUP BY idCredito