-- Modelo analítico para análisis de efectividad por banco
-- mart_efectividad_bancos.sql



WITH cobros_enriquecidos AS (
    SELECT * FROM "credifiel"."analytics"."int_cobros_enriquecidos"
)

SELECT
    idBanco,
    nombre_banco,
    COUNT(*) AS total_intentos,
    SUM(cobro_exitoso) AS total_exitosos,
    SUM(cobro_exitoso)::FLOAT / COUNT(*) AS tasa_exito,
    SUM(montoCobrado) AS monto_total_recuperado,
    SUM(montoCobrar) AS monto_total_intentado,
    SUM(montoCobrado)::FLOAT / NULLIF(SUM(montoCobrar), 0) AS tasa_recuperacion_monto,
    AVG(CASE WHEN cobro_exitoso = 1 THEN dias_hasta_cobro END) AS promedio_dias_hasta_cobro,
    -- Análisis por año
    SUM(CASE WHEN anio_origen = '2022' THEN cobro_exitoso ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN anio_origen = '2022' THEN 1 ELSE 0 END), 0) AS tasa_exito_2022,
    SUM(CASE WHEN anio_origen = '2023' THEN cobro_exitoso ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN anio_origen = '2023' THEN 1 ELSE 0 END), 0) AS tasa_exito_2023,
    SUM(CASE WHEN anio_origen = '2024' THEN cobro_exitoso ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN anio_origen = '2024' THEN 1 ELSE 0 END), 0) AS tasa_exito_2024,
    SUM(CASE WHEN anio_origen = '2025' THEN cobro_exitoso ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN anio_origen = '2025' THEN 1 ELSE 0 END), 0) AS tasa_exito_2025
FROM cobros_enriquecidos
GROUP BY idBanco, nombre_banco
ORDER BY tasa_exito DESC