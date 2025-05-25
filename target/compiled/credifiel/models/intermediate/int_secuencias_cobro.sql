-- Modelo intermedio para secuencias de cobro
-- int_secuencias_cobro.sql



SELECT
    idCredito,
    idBanco,
    idRespuestaBanco,
    fechaEnvioCobro,
    ROW_NUMBER() OVER(PARTITION BY idCredito ORDER BY fechaEnvioCobro) AS intento_numero,
    LEAD(idRespuestaBanco) OVER(PARTITION BY idCredito ORDER BY fechaEnvioCobro) AS siguiente_respuesta,
    cobro_exitoso
FROM "credifiel"."analytics"."int_cobros_enriquecidos"