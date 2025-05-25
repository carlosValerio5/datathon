-- Modelo intermedio que enriquece los cobros con información de bancos y respuestas
-- int_cobros_enriquecidos.sql



WITH cobros AS (
    SELECT * FROM "credifiel"."analytics"."int_cobros_consolidados"
),

bancos AS (
    SELECT * FROM "credifiel"."analytics"."stg_cat_banco"
),

respuestas AS (
    SELECT * FROM "credifiel"."analytics"."stg_cat_respuesta_bancos"
),

listas_cobro AS (
    SELECT * FROM "credifiel"."analytics"."stg_lista_cobro"
)

SELECT
    c.idListaCobro,
    c.idCredito,
    c.consecutivoCobro,
    c.idBanco,
    b.Nombre AS nombre_banco,
    c.montoExigible,
    c.montoCobrar,
    c.montoCobrado,
    c.fechaCobroBanco,
    c.idRespuestaBanco,
    r.Descripcion AS descripcion_respuesta,
    l.fechaCreacionLista,
    l.fechaEnvioCobro,
    c.anio_origen,
    -- Indicadores de éxito y eficiencia
CASE WHEN c.idRespuestaBanco = '00' THEN 1 ELSE 0 END AS cobro_exitoso,
CASE
  WHEN c.montoCobrado > 0 AND c.montoCobrar != 0 THEN c.montoCobrado / c.montoCobrar
  ELSE 0
END AS tasa_recuperacion,
CASE
  WHEN c.fechaCobroBanco IS NOT NULL AND l.fechaEnvioCobro IS NOT NULL THEN
    DATEDIFF('day', l.fechaEnvioCobro, c.fechaCobroBanco)
  ELSE NULL
END AS dias_hasta_cobro

FROM cobros c
LEFT JOIN bancos b ON c.idBanco = b.IdBanco
LEFT JOIN respuestas r ON c.idRespuestaBanco = r.IdRespuestaBanco
LEFT JOIN listas_cobro l ON c.idListaCobro = l.idListaCobro