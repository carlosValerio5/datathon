-- Modelo intermedio que consolida todos los datos de cobro de diferentes años
-- int_cobros_consolidados.sql



WITH cobros_2022 AS (
    SELECT
        idListaCobro,
        idCredito,
        consecutivoCobro,
        idBanco,
        montoExigible,
        montoCobrar,
        montoCobrado,
        fechaCobroBanco,
        idRespuestaBanco,
        '2022' AS anio_origen
    FROM "credifiel"."analytics"."stg_lista_cobro_detalle_2022"
),

cobros_2023 AS (
    SELECT
        idListaCobro,
        idCredito,
        consecutivoCobro,
        idBanco,
        montoExigible,
        montoCobrar,
        montoCobrado,
        fechaCobroBanco,
        idRespuestaBanco,
        '2023' AS anio_origen
    FROM "credifiel"."analytics"."stg_lista_cobro_detalle_2023"
),

cobros_2024 AS (
    SELECT
        idListaCobro,
        idCredito,
        consecutivoCobro,
        idBanco,
        montoExigible,
        montoCobrar,
        montoCobrado,
        fechaCobroBanco,
        idRespuestaBanco,
        '2024' AS anio_origen
    FROM "credifiel"."analytics"."stg_lista_cobro_detalle_2024"
),

cobros_2025 AS (
    SELECT
        idListaCobro,
        idCredito,
        consecutivoCobro,
        idBanco,
        montoExigible,
        montoCobrar,
        montoCobrado,
        fechaCobroBanco,
        idRespuestaBanco,
        '2025' AS anio_origen
    FROM "credifiel"."analytics"."stg_lista_cobro_detalle_2025"
)

SELECT * FROM cobros_2022
UNION ALL
SELECT * FROM cobros_2023
UNION ALL
SELECT * FROM cobros_2024
UNION ALL
SELECT * FROM cobros_2025